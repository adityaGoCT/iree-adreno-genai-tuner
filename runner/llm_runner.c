// On-device LLM runner + scheduler for IREE-compiled Llama modules.
//
// The module (scripts/llm/export_llama.py) exports static-shape entry points
//   prefill_c{C}_l{L}(tokens[1,C] i32, start[1] i32, last[1] i32, k0, v0, ... k{N-1}, v{N-1}) -> next[1] i32
//   decode_l{L}      (tokens[1,1] i32, start[1] i32, last[1] i32, k0, v0, ...)                -> next[1] i32
// where k_i/v_i are the KV cache tensors [max_ctx, kv_heads, head_dim] f16, updated in place, and
// L is the KV "bucket": attention only reads cache[:L]. The runner owns the cache buffers on the
// device for the whole generation, so no KV data ever crosses the host/device boundary.
//
// Scheduler (--schedule):
//   bucketed  each step uses the smallest compiled bucket L >= tokens-in-cache-after-step
//   max       always the largest bucket (baseline: what a single static-shape graph would do)
// Prefill is split into chunks of --chunk tokens; the last chunk is padded.
//
// Output: one JSON object on stdout with token ids and timings.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/context_util.h"
#include "iree/vm/api.h"

// IREE >= 3.12 returns a device list (+ optional replay recorder) from context creation.
#if __has_include("iree/hal/replay/recorder.h")
#define IREE_312_API 1
#include "iree/hal/replay/recorder.h"
#endif

IREE_FLAG(string, prompt, "", "File with whitespace-separated prompt token ids.");
IREE_FLAG(int32_t, max_new_tokens, 32, "Tokens to generate after the prompt.");
IREE_FLAG(int32_t, chunk, 128, "Prefill chunk size (must match exported prefill_c<chunk>_*).");
IREE_FLAG(string, buckets, "256,512,1024,2048,4096", "Compiled KV-length buckets, ascending.");
IREE_FLAG(string, schedule, "bucketed", "bucketed | max");
IREE_FLAG(int32_t, n_layers, 28, "Transformer layers (KV cache tensors = 2 * n_layers).");
IREE_FLAG(int32_t, n_kv_heads, 8, "KV heads.");
IREE_FLAG(int32_t, head_dim, 128, "Head dim.");
IREE_FLAG(int32_t, max_ctx, 4096, "KV cache capacity (tokens).");
IREE_FLAG(string, stop_ids, "", "Comma-separated stop token ids (empty = never stop early).");
IREE_FLAG(int32_t, warmup, 1, "Untimed warm-up calls per used function (pipeline creation).");

#define MAX_BUCKETS 16
#define MAX_TOKENS 8192

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int parse_ints(const char* s, int32_t* out, int max) {
  int n = 0;
  while (s && *s && n < max) {
    char* end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s) { ++s; continue; }
    out[n++] = (int32_t)v;
    s = end;
  }
  return n;
}

typedef struct {
  iree_vm_context_t* context;
  iree_vm_module_t* module;
  iree_hal_device_t* device;
  iree_hal_allocator_t* allocator;
  iree_hal_buffer_view_t** caches;  // 2 * n_layers
  int n_caches;
} runner_t;

static iree_status_t make_i32(runner_t* r, const int32_t* data, iree_host_size_t n,
                              iree_hal_dim_t d0, iree_hal_dim_t d1, int rank,
                              iree_hal_buffer_view_t** out) {
  iree_hal_dim_t shape[2] = {d0, d1};
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  return iree_hal_buffer_view_allocate_buffer_copy(
      r->device, r->allocator, rank, shape, IREE_HAL_ELEMENT_TYPE_INT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
      iree_make_const_byte_span(data, n * sizeof(int32_t)), out);
}

// Calls `fn` with (tokens[1,C], start, last, caches...) and returns the next token id.
static iree_status_t call_step(runner_t* r, const char* fn, const int32_t* tokens, int C,
                               int32_t start, int32_t last, int32_t* out_next) {
  iree_vm_function_t function;
  IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_name(
      r->module, IREE_VM_FUNCTION_LINKAGE_EXPORT, iree_make_cstring_view(fn), &function),
      "looking up %s", fn);
  iree_allocator_t host = iree_allocator_system();
  iree_vm_list_t* inputs = NULL;
  iree_vm_list_t* outputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                           3 + r->n_caches, host, &inputs));
  iree_status_t status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 1, host, &outputs);

  iree_hal_buffer_view_t *tok_v = NULL, *start_v = NULL, *last_v = NULL;
  if (iree_status_is_ok(status)) status = make_i32(r, tokens, C, 1, C, 2, &tok_v);
  if (iree_status_is_ok(status)) status = make_i32(r, &start, 1, 1, 0, 1, &start_v);
  if (iree_status_is_ok(status)) status = make_i32(r, &last, 1, 1, 0, 1, &last_v);
  if (iree_status_is_ok(status)) {
    iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(tok_v);
    status = iree_vm_list_push_ref_move(inputs, &ref);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(start_v);
    status = iree_vm_list_push_ref_move(inputs, &ref);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(last_v);
    status = iree_vm_list_push_ref_move(inputs, &ref);
  }
  for (int i = 0; iree_status_is_ok(status) && i < r->n_caches; ++i) {
    iree_vm_ref_t ref = iree_hal_buffer_view_retain_ref(r->caches[i]);
    status = iree_vm_list_push_ref_move(inputs, &ref);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(r->context, function, IREE_VM_INVOCATION_FLAG_NONE,
                            /*policy=*/NULL, inputs, outputs, host);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_view_t* res = iree_vm_list_get_buffer_view_assign(outputs, 0);
    if (!res) {
      status = iree_make_status(IREE_STATUS_INTERNAL, "%s returned no buffer view", fn);
    } else {
      status = iree_hal_device_transfer_d2h(
          r->device, iree_hal_buffer_view_buffer(res), 0, out_next, sizeof(int32_t),
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    }
  }
  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  return status;
}

static int pick_bucket(const int32_t* buckets, int nb, int needed, int use_max) {
  if (use_max) return buckets[nb - 1];
  for (int i = 0; i < nb; ++i) {
    if (buckets[i] >= needed) return buckets[i];
  }
  return -1;
}

static iree_status_t run(runner_t* r) {
  static int32_t prompt[MAX_TOKENS];
  static int32_t generated[MAX_TOKENS];
  int32_t buckets[MAX_BUCKETS];
  int32_t stops[16];
  int nb = parse_ints(FLAG_buckets, buckets, MAX_BUCKETS);
  int ns = parse_ints(FLAG_stop_ids, stops, 16);
  int use_max = strcmp(FLAG_schedule, "max") == 0;

  FILE* f = fopen(FLAG_prompt, "r");
  if (!f) return iree_make_status(IREE_STATUS_NOT_FOUND, "cannot open --prompt=%s", FLAG_prompt);
  static char text[MAX_TOKENS * 12];
  size_t len = fread(text, 1, sizeof(text) - 1, f);
  fclose(f);
  text[len] = 0;
  int n_prompt = parse_ints(text, prompt, MAX_TOKENS);
  int C = FLAG_chunk;
  if (n_prompt == 0) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty prompt");
  int total = n_prompt + FLAG_max_new_tokens;
  if (total > FLAG_max_ctx || ((n_prompt + C - 1) / C) * C > FLAG_max_ctx) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "prompt %d + new %d exceeds max_ctx %d",
                            n_prompt, FLAG_max_new_tokens, FLAG_max_ctx);
  }

  // Warm-up: the first call of each function creates its Vulkan pipelines; keep that out of the
  // timings. Simulate the schedule to find exactly the functions it will use. Warm-up calls write
  // garbage KV at positions [0, C) which the real prefill overwrites before anything reads them.
  char fn[64];
  int32_t dummy = 0, next = 0;
  static int32_t zeros[MAX_TOKENS];
  int used_prefill[MAX_BUCKETS] = {0}, used_decode[MAX_BUCKETS] = {0};
  for (int start = 0; start < n_prompt; start += C) {
    int L = pick_bucket(buckets, nb, start + C, use_max);
    for (int i = 0; i < nb; ++i) if (buckets[i] == L) used_prefill[i] = 1;
  }
  for (int pos = n_prompt; pos < total - 1; ++pos) {
    int L = pick_bucket(buckets, nb, pos + 1, use_max);
    for (int i = 0; i < nb; ++i) if (buckets[i] == L) used_decode[i] = 1;
  }
  double t_warm = now_ms();
  for (int w = 0; w < FLAG_warmup; ++w) {
    for (int i = 0; i < nb; ++i) {
      if (used_prefill[i]) {
        snprintf(fn, sizeof(fn), "prefill_c%d_l%d", C, buckets[i]);
        IREE_RETURN_IF_ERROR(call_step(r, fn, zeros, C, 0, 0, &dummy));
      }
      if (used_decode[i]) {
        snprintf(fn, sizeof(fn), "decode_l%d", buckets[i]);
        IREE_RETURN_IF_ERROR(call_step(r, fn, zeros, 1, 0, 0, &dummy));
      }
    }
  }
  t_warm = now_ms() - t_warm;

  // ---- Prefill (chunked) ----
  static double chunk_ms[MAX_TOKENS];
  static int chunk_bucket[MAX_TOKENS];
  int n_chunks = 0;
  double t0 = now_ms();
  for (int start = 0; start < n_prompt; start += C) {
    int n = n_prompt - start < C ? n_prompt - start : C;
    static int32_t buf[MAX_TOKENS];
    memset(buf, 0, sizeof(int32_t) * C);
    memcpy(buf, prompt + start, sizeof(int32_t) * n);
    int L = pick_bucket(buckets, nb, start + C, use_max);
    if (L < 0) return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "no bucket >= %d", start + C);
    snprintf(fn, sizeof(fn), "prefill_c%d_l%d", C, L);
    double tc = now_ms();
    IREE_RETURN_IF_ERROR(call_step(r, fn, buf, C, start, n - 1, &next));
    chunk_ms[n_chunks] = now_ms() - tc;
    chunk_bucket[n_chunks++] = L;
  }
  double prefill_ms = now_ms() - t0;

  // ---- Decode ----
  static double step_ms[MAX_TOKENS];
  static int step_bucket[MAX_TOKENS];
  int n_gen = 0;
  generated[n_gen++] = next;  // first token comes from prefill
  double t1 = now_ms();
  int pos = n_prompt;
  int stopped = 0;
  while (n_gen < FLAG_max_new_tokens) {
    for (int s = 0; s < ns; ++s) if (next == stops[s]) stopped = 1;
    if (stopped) break;
    int L = pick_bucket(buckets, nb, pos + 1, use_max);
    if (L < 0) break;
    snprintf(fn, sizeof(fn), "decode_l%d", L);
    double ts = now_ms();
    IREE_RETURN_IF_ERROR(call_step(r, fn, &next, 1, pos, 0, &next));
    step_ms[n_gen - 1] = now_ms() - ts;
    step_bucket[n_gen - 1] = L;
    generated[n_gen++] = next;
    ++pos;
  }
  double decode_ms = now_ms() - t1;
  int n_steps = n_gen - 1;

  // ---- Report ----
  printf("{\"schedule\":\"%s\",\"chunk\":%d,\"prompt_tokens\":%d,\"new_tokens\":%d,", FLAG_schedule,
         C, n_prompt, n_gen);
  printf("\"warmup_ms\":%.1f,\"prefill_ms\":%.1f,\"prefill_tok_s\":%.2f,\"ttft_ms\":%.1f,", t_warm,
         prefill_ms, n_prompt / (prefill_ms / 1e3), prefill_ms);
  printf("\"decode_ms\":%.1f,\"decode_steps\":%d,\"decode_tok_s\":%.2f,", decode_ms, n_steps,
         n_steps > 0 ? n_steps / (decode_ms / 1e3) : 0.0);
  printf("\"chunks\":[");
  for (int i = 0; i < n_chunks; ++i) printf("%s[%d,%.1f]", i ? "," : "", chunk_bucket[i], chunk_ms[i]);
  printf("],\"steps\":[");
  for (int i = 0; i < n_steps; ++i) printf("%s[%d,%.2f]", i ? "," : "", step_bucket[i], step_ms[i]);
  printf("],\"tokens\":[");
  for (int i = 0; i < n_gen; ++i) printf("%s%d", i ? "," : "", generated[i]);
  printf("]}\n");
  fflush(stdout);
  return iree_ok_status();
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  iree_allocator_t host = iree_allocator_system();
  runner_t r = {0};

  iree_vm_instance_t* instance = NULL;
  iree_status_t status = iree_tooling_create_instance(host, &instance);
  iree_tooling_module_list_t modules;
  iree_tooling_module_list_initialize(&modules);
  if (iree_status_is_ok(status)) status = iree_tooling_load_modules_from_flags(instance, host, &modules);
  if (iree_status_is_ok(status)) {
    r.module = iree_tooling_module_list_back(&modules);
    iree_vm_module_retain(r.module);
#if IREE_312_API
    iree_hal_device_list_t* device_list = NULL;
    iree_hal_replay_recorder_t* recorder = NULL;
    status = iree_tooling_create_context_from_flags(instance, modules.count, modules.values,
                                                    iree_make_cstring_view("vulkan"), host,
                                                    &r.context, &device_list, &r.allocator,
                                                    &recorder);
    if (iree_status_is_ok(status)) {
      r.device = iree_hal_device_list_at(device_list, 0);
      iree_hal_device_retain(r.device);
    }
    if (device_list) iree_hal_device_list_free(device_list);
    if (recorder) iree_hal_replay_recorder_release(recorder);
#else
    status = iree_tooling_create_context_from_flags(instance, modules.count, modules.values,
                                                    iree_make_cstring_view("vulkan"), host,
                                                    &r.context, &r.device, &r.allocator);
#endif
  }
  iree_tooling_module_list_reset(&modules);

  // KV caches: zero-initialised device-local buffers kept for the whole run.
  r.n_caches = 2 * FLAG_n_layers;
  r.caches = calloc(r.n_caches, sizeof(*r.caches));
  iree_host_size_t cache_bytes = (iree_host_size_t)FLAG_max_ctx * FLAG_n_kv_heads * FLAG_head_dim * 2;
  void* zeros = calloc(1, cache_bytes);
  for (int i = 0; iree_status_is_ok(status) && i < r.n_caches; ++i) {
    iree_hal_dim_t shape[3] = {FLAG_max_ctx, FLAG_n_kv_heads, FLAG_head_dim};
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
    status = iree_hal_buffer_view_allocate_buffer_copy(
        r.device, r.allocator, 3, shape, IREE_HAL_ELEMENT_TYPE_FLOAT_16,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
        iree_make_const_byte_span(zeros, cache_bytes), &r.caches[i]);
  }
  free(zeros);

  if (iree_status_is_ok(status)) status = run(&r);

  for (int i = 0; i < r.n_caches; ++i) iree_hal_buffer_view_release(r.caches[i]);
  free(r.caches);
  iree_hal_allocator_release(r.allocator);
  iree_hal_device_release(r.device);
  iree_vm_context_release(r.context);
  iree_vm_module_release(r.module);
  iree_vm_instance_release(instance);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }
  return 0;
}
