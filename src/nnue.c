#include <stdio.h>
#include <stdlib.h>
#if defined(__AVX2__)
    #include <immintrin.h>
    #if defined(__FMA__)
        #define NNUE_FMADD(a, b, c) _mm256_fmadd_ps((a), (b), (c))
    #else
        #define NNUE_FMADD(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
    #endif
    // int8 dot-accumulate (uint8 x int8 -> int32), VNNI if available.
    #if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        #define NNUE_DPBUSD(acc, a, b) _mm256_dpbusd_epi32((acc), (a), (b))
        #define NNUE_HAVE_VNNI 1
    #elif defined(__AVXVNNI__)
        #define NNUE_DPBUSD(acc, a, b) _mm256_dpbusd_avx_epi32((acc), (a), (b))
        #define NNUE_HAVE_VNNI 1
    #else
        // AVX2 fallback (no VNNI): uint8 x int8 -> int16 pairwise sums (maddubs),
        // then int16 pairs -> int32 (madd). Activations are clamped to [0,127] and
        // weights are int8, so each maddubs pairwise sum (<= 2*127*127 = 32258) stays
        // within int16 range. Numerically identical to the scalar/VNNI paths.
        static inline __m256i nnue_dpbusd_avx2(__m256i acc, __m256i a, __m256i b) {
            __m256i p = _mm256_maddubs_epi16(a, b);
            p = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
            return _mm256_add_epi32(acc, p);
        }
        #define NNUE_DPBUSD(acc, a, b) nnue_dpbusd_avx2((acc), (a), (b))
    #endif
    // int8 dot-accumulate is available in SIMD form (VNNI or the AVX2 fallback).
    #define NNUE_HAVE_DP 1
    static inline int nnue_hsum_i32(__m256i v) {
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
        return _mm_cvtsi128_si32(s);
    }
#endif
#include "nnue.h"
#include "board.h"

// Definitions for globals declared extern in nnue.h
NNUE* nnue;
char fen[MAX_FEN_LENGTH];

// Feature index, matching bullet's Chess768 encoding, offset by the king bucket:
//   index = bucket*768 + color*384 + pieceType*64 + sq   (white/direct)
//   index = bucket*768 + (!color)*384 + pieceType*64 + (sq^56)  (black/perspective)
// (piece is 1..6 here, so piece-1 == bullet's 0..5 pieceType.) bucket is 0 in the
// default (non-KB) build.
int getInputIndexOf(int bucket, int color, int piece, int sq) {
    return bucket * INPUTS_COUNT + color * 64 * 6 + (piece - 1) * 64 + sq;
}

// King-bucket map by king square. Must match kb_bucket() in the bullet trainer
// (nnue/trainer/src/main.rs). Counts: 4 (2x2 quadrants), 8 (4 files x 2 ranks),
// 16 (4 files x 4 ranks).
int nnueKingBucket(int sq) {
#ifdef NNUE_KB
    int rank = sq / 8, file = sq % 8;
    #if NNUE_KB == 4
        return (rank / 4) * 2 + (file / 4);
    #elif NNUE_KB == 8
        return (rank / 4) * 4 + (file / 2);
    #elif NNUE_KB == 16
        return (rank / 2) * 4 + (file / 2);
    #else
        #error "unsupported NNUE_KB (use 4, 8, or 16)"
    #endif
#else
    (void) sq;
    return 0;
#endif
}

static void addFeature(S32* acc, const S16* col) {
#if defined(__AVX2__)
    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
        __m256i a = _mm256_loadu_si256((const __m256i*) &acc[i]);
        __m256i c = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*) &col[i]));
        _mm256_storeu_si256((__m256i*) &acc[i], _mm256_add_epi32(a, c));
    }
#else
    for (int i = 0; i < INNER_LAYER_COUNT; ++i)
        acc[i] += col[i];
#endif
}

static void subFeature(S32* acc, const S16* col) {
#if defined(__AVX2__)
    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
        __m256i a = _mm256_loadu_si256((const __m256i*) &acc[i]);
        __m256i c = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*) &col[i]));
        _mm256_storeu_si256((__m256i*) &acc[i], _mm256_sub_epi32(a, c));
    }
#else
    for (int i = 0; i < INNER_LAYER_COUNT; ++i)
        acc[i] -= col[i];
#endif
}

void setDirectNNUEInput(NNUE* nnue, int index) {
    if (nnue->inputs[index])
        return;
    nnue->inputs[index] = 1;
    addFeature(nnue->accumulators, nnue->featureWeights[index]);
}

void resetDirectNNUEInput(NNUE* nnue, int index) {
    if (!nnue->inputs[index])
        return;
    nnue->inputs[index] = 0;
    subFeature(nnue->accumulators, nnue->featureWeights[index]);
}

void setPerspectiveNNUEInput(NNUE* nnue, int index) {
    if (nnue->inputs_perspective[index])
        return;
    nnue->inputs_perspective[index] = 1;
    addFeature(nnue->accumulators_perspective, nnue->featureWeights[index]);
}

void resetPerspectiveNNUEInput(NNUE* nnue, int index) {
    if (!nnue->inputs_perspective[index])
        return;
    nnue->inputs_perspective[index] = 0;
    subFeature(nnue->accumulators_perspective, nnue->featureWeights[index]);
}

void resetNNUE(NNUE* nnue) {
    nnue->wkBucket = 0;
    nnue->bkBucket = 0;
    for (int i = 0; i < NNUE_FT_INPUTS; ++i) {
        nnue->inputs[i] = 0;
        nnue->inputs_perspective[i] = 0;
    }
    // Accumulators start from the feature bias so incremental updates just
    // add/subtract feature columns afterwards.
    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        nnue->accumulators[i] = nnue->featureBias[i];
        nnue->accumulators_perspective[i] = nnue->featureBias[i];
    }
    nnue->eval = 0;
}

static inline S32 screlu(S32 x) {
    if (x < 0)
        x = 0;
    else if (x > NNUE_QA)
        x = NNUE_QA;
    return x * x;
}

#ifdef NNUE_L2
// Feature-transformer element -> real SCReLU: clamp(acc/QA, 0, 1)^2.
// Multiply by the reciprocal (folded to a constant) instead of dividing.
static inline float screluf(S32 acc) {
    float x = acc < 0 ? 0.0f : (acc > NNUE_QA ? (float) NNUE_QA : (float) acc);
    x *= (1.0f / (float) NNUE_QA);
    return x * x;
}
// Hidden-layer SCReLU on real values: clamp(x, 0, 1)^2.
static inline float screluf01(float x) {
    if (x < 0.0f) x = 0.0f;
    else if (x > 1.0f) x = 1.0f;
    return x * x;
}
#endif

#ifdef NNUE_L2_I8
static inline signed char q_i8(float v, float scale) {
    int q = (int) (v * scale + (v >= 0.0f ? 0.5f : -0.5f));
    return (signed char) (q < -127 ? -127 : (q > 127 ? 127 : q));
}
static inline int q_i32(float v, float scale) {
    return (int) (v * scale + (v >= 0.0f ? 0.5f : -0.5f));
}
#if defined(__AVX2__)
// 32 accumulator values -> 32 uint8 activations: round(SCReLU(acc)*127), in [0,127].
static inline void ft_to_u8(const S32* acc, unsigned char* dst) {
    const __m256  inv   = _mm256_set1_ps(1.0f / (float) NNUE_QA);
    const __m256  act   = _mm256_set1_ps((float) NNUE_I8_ACT);
    const __m256i zeroi = _mm256_setzero_si256();
    const __m256i qai   = _mm256_set1_epi32(NNUE_QA);
    __m256i r[4];
    for (int k = 0; k < 4; ++k) {
        __m256i v = _mm256_min_epi32(_mm256_max_epi32(
            _mm256_loadu_si256((const __m256i*) (acc + 8 * k)), zeroi), qai);
        __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(v), inv);
        f = _mm256_mul_ps(_mm256_mul_ps(f, f), act);   // SCReLU * 127, in [0,127]
        r[k] = _mm256_cvtps_epi32(f);                  // round to nearest
    }
    __m256i p = _mm256_packus_epi16(_mm256_packus_epi32(r[0], r[1]),
                                    _mm256_packus_epi32(r[2], r[3]));
    p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    _mm256_storeu_si256((__m256i*) dst, p);
}
#endif
#endif

// One perspective's SCReLU dot product: sum_i screlu(acc[i]) * w[i], in i64.
static long long dotScrelu(const S32* acc, const S16* w) {
#if defined(__AVX2__)
    __m256i sum = _mm256_setzero_si256();          // 4 x i64
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa = _mm256_set1_epi32(NNUE_QA);
    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
        __m256i a = _mm256_loadu_si256((const __m256i*) &acc[i]);
        a = _mm256_min_epi32(_mm256_max_epi32(a, zero), qa);   // clamp [0, QA]
        a = _mm256_mullo_epi32(a, a);                          // square (<= QA*QA)
        __m256i wi = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*) &w[i]));
        __m256i prod = _mm256_mullo_epi32(a, wi);              // 8 x i32
        // widen to i64 and accumulate to avoid overflow (matches scalar)
        sum = _mm256_add_epi64(sum, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod)));
        sum = _mm256_add_epi64(sum, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod, 1)));
    }
    long long t[4];
    _mm256_storeu_si256((__m256i*) t, sum);
    return t[0] + t[1] + t[2] + t[3];
#else
    long long s = 0;
    for (int i = 0; i < INNER_LAYER_COUNT; ++i)
        s += (long long) screlu(acc[i]) * w[i];
    return s;
#endif
}

// Mirrors bullet's Network::evaluate exactly (see bullet/examples/simple.rs).
// Returns a side-to-move-relative score in centipawns.
void recalculateEval(NNUE* nnue, int color) {
    const S32* us   = (color == WHITE) ? nnue->accumulators : nnue->accumulators_perspective;
    const S32* them = (color == WHITE) ? nnue->accumulators_perspective : nnue->accumulators;

#ifdef NNUE_L2
    float out;
#if defined(NNUE_L2_I8)
    // ===== int8 head: uint8 activations x int8 weights =====
    unsigned char hq[2 * INNER_LAYER_COUNT];
#if defined(__AVX2__) && (INNER_LAYER_COUNT % 32 == 0)
    for (int i = 0; i < INNER_LAYER_COUNT; i += 32) {
        ft_to_u8(&us[i],   &hq[i]);
        ft_to_u8(&them[i], &hq[INNER_LAYER_COUNT + i]);
    }
#else
    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        int qu = (int) (screluf(us[i])   * NNUE_I8_ACT + 0.5f);
        int qt = (int) (screluf(them[i]) * NNUE_I8_ACT + 0.5f);
        hq[i]                     = (unsigned char) (qu > 127 ? 127 : qu);
        hq[INNER_LAYER_COUNT + i] = (unsigned char) (qt > 127 ? 127 : qt);
    }
#endif
    int a[NNUE_L2];
#if defined(NNUE_HAVE_DP) && (NNUE_L2 % 4 == 0)
    // 4 outputs at a time with independent accumulators (hides dpbusd latency);
    // hq chunk loaded once and reused across the 4 columns.
    for (int j = 0; j < NNUE_L2; j += 4) {
        const signed char* c0 = &nnue->l1w_i8[(size_t) (j + 0) * (2 * INNER_LAYER_COUNT)];
        const signed char* c1 = &nnue->l1w_i8[(size_t) (j + 1) * (2 * INNER_LAYER_COUNT)];
        const signed char* c2 = &nnue->l1w_i8[(size_t) (j + 2) * (2 * INNER_LAYER_COUNT)];
        const signed char* c3 = &nnue->l1w_i8[(size_t) (j + 3) * (2 * INNER_LAYER_COUNT)];
        __m256i s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
        __m256i s2 = _mm256_setzero_si256(), s3 = _mm256_setzero_si256();
        for (int i = 0; i < 2 * INNER_LAYER_COUNT; i += 32) {
            const __m256i h = _mm256_loadu_si256((const __m256i*) &hq[i]);
            s0 = NNUE_DPBUSD(s0, h, _mm256_loadu_si256((const __m256i*) &c0[i]));
            s1 = NNUE_DPBUSD(s1, h, _mm256_loadu_si256((const __m256i*) &c1[i]));
            s2 = NNUE_DPBUSD(s2, h, _mm256_loadu_si256((const __m256i*) &c2[i]));
            s3 = NNUE_DPBUSD(s3, h, _mm256_loadu_si256((const __m256i*) &c3[i]));
        }
        a[j + 0] = nnue_hsum_i32(s0) + nnue->l1b_i32[j + 0];
        a[j + 1] = nnue_hsum_i32(s1) + nnue->l1b_i32[j + 1];
        a[j + 2] = nnue_hsum_i32(s2) + nnue->l1b_i32[j + 2];
        a[j + 3] = nnue_hsum_i32(s3) + nnue->l1b_i32[j + 3];
    }
#elif defined(NNUE_HAVE_DP)
    for (int j = 0; j < NNUE_L2; ++j) {
        const signed char* col = &nnue->l1w_i8[(size_t) j * (2 * INNER_LAYER_COUNT)];
        __m256i acc = _mm256_setzero_si256();
        for (int i = 0; i < 2 * INNER_LAYER_COUNT; i += 32)
            acc = NNUE_DPBUSD(acc,
                _mm256_loadu_si256((const __m256i*) &hq[i]),
                _mm256_loadu_si256((const __m256i*) &col[i]));
        a[j] = nnue_hsum_i32(acc) + nnue->l1b_i32[j];
    }
#else
    for (int j = 0; j < NNUE_L2; ++j) {
        const signed char* col = &nnue->l1w_i8[(size_t) j * (2 * INNER_LAYER_COUNT)];
        int s = nnue->l1b_i32[j];
        for (int i = 0; i < 2 * INNER_LAYER_COUNT; ++i) s += (int) hq[i] * col[i];
        a[j] = s;
    }
#endif
    unsigned char aq[NNUE_L2];
    for (int j = 0; j < NNUE_L2; ++j) {
        float r = screluf01((float) a[j] / (float) NNUE_I8_MUL);
        int q = (int) (r * NNUE_I8_ACT + 0.5f);
        aq[j] = (unsigned char) (q > 127 ? 127 : q);
    }
    int o;
#if defined(NNUE_HAVE_DP) && (NNUE_L2 == 32)
    {
        __m256i acc = NNUE_DPBUSD(_mm256_setzero_si256(),
            _mm256_loadu_si256((const __m256i*) aq),
            _mm256_loadu_si256((const __m256i*) nnue->l2w_i8));
        o = nnue_hsum_i32(acc) + nnue->l2b_i32;
    }
#else
    o = nnue->l2b_i32;
    for (int j = 0; j < NNUE_L2; ++j) o += (int) aq[j] * nnue->l2w_i8[j];
#endif
    out = (float) o / (float) NNUE_I8_MUL;
#else
    // FT(i16) -> SCReLU -> l1(2*INNER -> L2) -> SCReLU -> l2(L2 -> 1), in float.
    float hidden[2 * INNER_LAYER_COUNT];
#if defined(__AVX2__) && (NNUE_L2 == 32) && (INNER_LAYER_COUNT % 8 == 0)
    // FT SCReLU -> hidden, vectorised: clamp(acc,0,QA)*1/QA, squared.
    {
        const __m256  inv   = _mm256_set1_ps(1.0f / (float) NNUE_QA);
        const __m256i zeroi = _mm256_setzero_si256();
        const __m256i qai   = _mm256_set1_epi32(NNUE_QA);
        for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
            __m256i u = _mm256_min_epi32(_mm256_max_epi32(
                _mm256_loadu_si256((const __m256i*) &us[i]), zeroi), qai);
            __m256 uf = _mm256_mul_ps(_mm256_cvtepi32_ps(u), inv);
            _mm256_storeu_ps(&hidden[i], _mm256_mul_ps(uf, uf));
            __m256i t = _mm256_min_epi32(_mm256_max_epi32(
                _mm256_loadu_si256((const __m256i*) &them[i]), zeroi), qai);
            __m256 tf = _mm256_mul_ps(_mm256_cvtepi32_ps(t), inv);
            _mm256_storeu_ps(&hidden[INNER_LAYER_COUNT + i], _mm256_mul_ps(tf, tf));
        }
    }
    // l1w is input-major [i*NNUE_L2 + j] (bullet stores affine as [input][output]).
    // 32 outputs held in 4 AVX vectors; branchless (misprediction costs more than
    // the extra FMAs on the ~half of inputs that SCReLU zeroed).
    __m256 a0 = _mm256_loadu_ps(&nnue->l1b[0]);
    __m256 a1 = _mm256_loadu_ps(&nnue->l1b[8]);
    __m256 a2 = _mm256_loadu_ps(&nnue->l1b[16]);
    __m256 a3 = _mm256_loadu_ps(&nnue->l1b[24]);
    for (int i = 0; i < 2 * INNER_LAYER_COUNT; ++i) {
        const __m256 hv = _mm256_set1_ps(hidden[i]);
        const float* r = &nnue->l1w[(size_t) i * NNUE_L2];
        a0 = NNUE_FMADD(hv, _mm256_loadu_ps(r + 0),  a0);
        a1 = NNUE_FMADD(hv, _mm256_loadu_ps(r + 8),  a1);
        a2 = NNUE_FMADD(hv, _mm256_loadu_ps(r + 16), a2);
        a3 = NNUE_FMADD(hv, _mm256_loadu_ps(r + 24), a3);
    }
    // SCReLU on the 32 hidden units: clamp [0,1], square.
    const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    a0 = _mm256_min_ps(_mm256_max_ps(a0, zero), one); a0 = _mm256_mul_ps(a0, a0);
    a1 = _mm256_min_ps(_mm256_max_ps(a1, zero), one); a1 = _mm256_mul_ps(a1, a1);
    a2 = _mm256_min_ps(_mm256_max_ps(a2, zero), one); a2 = _mm256_mul_ps(a2, a2);
    a3 = _mm256_min_ps(_mm256_max_ps(a3, zero), one); a3 = _mm256_mul_ps(a3, a3);
    // l2: dot(acc, l2w) + l2b
    __m256 s = _mm256_mul_ps(a0, _mm256_loadu_ps(&nnue->l2w[0]));
    s = NNUE_FMADD(a1, _mm256_loadu_ps(&nnue->l2w[8]),  s);
    s = NNUE_FMADD(a2, _mm256_loadu_ps(&nnue->l2w[16]), s);
    s = NNUE_FMADD(a3, _mm256_loadu_ps(&nnue->l2w[24]), s);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    out = _mm_cvtss_f32(lo) + nnue->l2b;
#else
    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        hidden[i]                     = screluf(us[i]);
        hidden[INNER_LAYER_COUNT + i] = screluf(them[i]);
    }
    float acc[NNUE_L2];
    for (int j = 0; j < NNUE_L2; ++j)
        acc[j] = nnue->l1b[j];
    for (int i = 0; i < 2 * INNER_LAYER_COUNT; ++i) {
        const float h = hidden[i];
        if (h == 0.0f) continue;
        const float* row = &nnue->l1w[(size_t) i * NNUE_L2];
        for (int j = 0; j < NNUE_L2; ++j)
            acc[j] += h * row[j];
    }
    for (int j = 0; j < NNUE_L2; ++j)
        acc[j] = screluf01(acc[j]);
    out = nnue->l2b;
    for (int j = 0; j < NNUE_L2; ++j)
        out += acc[j] * nnue->l2w[j];
#endif
#endif  // NNUE_L2_I8 vs f32 head

    int e = (int) (out * NNUE_SCALE);
    if (e > NNUE_L2_EVAL_CAP) e = NNUE_L2_EVAL_CAP;
    else if (e < -NNUE_L2_EVAL_CAP) e = -NNUE_L2_EVAL_CAP;
    nnue->eval = e;
#else
    long long output = dotScrelu(us, nnue->outputWeights)
                     + dotScrelu(them, nnue->outputWeights + INNER_LAYER_COUNT);

    output /= NNUE_QA;                          // QA*QA*QB -> QA*QB
    output += nnue->outputBias;                 // QA*QB-quantised
    output *= NNUE_SCALE;
    output /= (long long) NNUE_QA * NNUE_QB;    // dequantise

    nnue->eval = (int) output;
#endif
}

// Loads a raw bullet-exported network (quantised i16, in save order:
// l0w, l0b, l1w, l1b). Returns 1 on success.
int loadBulletNet(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return 0;

    size_t got = 0;
    got += fread(nnue->featureWeights, sizeof(S16), (size_t) NNUE_FT_INPUTS * INNER_LAYER_COUNT, f);
    got += fread(nnue->featureBias,    sizeof(S16), INNER_LAYER_COUNT, f);
#ifdef NNUE_L2
    // save order: l0w, l0b (i16), then l1w, l1b, l2w, l2b (f32)
    got += fread(nnue->l1w, sizeof(float), (size_t) 2 * INNER_LAYER_COUNT * NNUE_L2, f);
    got += fread(nnue->l1b, sizeof(float), NNUE_L2, f);
    got += fread(nnue->l2w, sizeof(float), NNUE_L2, f);
    got += fread(&nnue->l2b, sizeof(float), 1, f);
    fclose(f);

    size_t expected = (size_t) NNUE_FT_INPUTS * INNER_LAYER_COUNT + INNER_LAYER_COUNT
                    + (size_t) 2 * INNER_LAYER_COUNT * NNUE_L2 + NNUE_L2 + NNUE_L2 + 1;
    if (got != expected)
        return 0;
  #ifdef NNUE_L2_I8
    // Post-training quantise the f32 head into int8 tables (weights x64, biases
    // x8128 to match one uint8*int8 affine's product scale). l1w_i8 is stored
    // OUTPUT-major [j*(2*INNER)+i] so each output's weights are contiguous for
    // the per-output dpbusd dot product.
    for (int i = 0; i < 2 * INNER_LAYER_COUNT; ++i)
        for (int j = 0; j < NNUE_L2; ++j)
            nnue->l1w_i8[(size_t) j * (2 * INNER_LAYER_COUNT) + i] =
                q_i8(nnue->l1w[(size_t) i * NNUE_L2 + j], NNUE_I8_W);
    for (int j = 0; j < NNUE_L2; ++j)
        nnue->l1b_i32[j] = q_i32(nnue->l1b[j], NNUE_I8_MUL);
    for (int j = 0; j < NNUE_L2; ++j)
        nnue->l2w_i8[j] = q_i8(nnue->l2w[j], NNUE_I8_W);
    nnue->l2b_i32 = q_i32(nnue->l2b, NNUE_I8_MUL);
  #endif
#else
    got += fread(nnue->outputWeights,  sizeof(S16), 2 * INNER_LAYER_COUNT, f);
    got += fread(&nnue->outputBias,    sizeof(S16), 1, f);
    fclose(f);

    size_t expected = (size_t) NNUE_FT_INPUTS * INNER_LAYER_COUNT
                    + INNER_LAYER_COUNT + 2 * INNER_LAYER_COUNT + 1;
    if (got != expected)
        return 0;
#endif

    resetNNUE(nnue);
    return 1;
}

#ifdef NNUE_KB
// Recompute one perspective's accumulator from scratch with its current bucket.
// Called when that side's king changed board quadrant (all its features re-index).
static void refreshSide(Board* board, int side) {
    if (side == WHITE) {
        for (int i = 0; i < INNER_LAYER_COUNT; ++i)
            nnue->accumulators[i] = nnue->featureBias[i];
        for (int i = 0; i < NNUE_FT_INPUTS; ++i)
            nnue->inputs[i] = 0;
        for (int sq = 0; sq < 64; ++sq) {
            int pc = board->squares[sq];
            if (!pc) continue;
            int idx = getInputIndexOf(nnue->wkBucket, pieceColor(pc), pieceType(pc), sq);
            nnue->inputs[idx] = 1;
            addFeature(nnue->accumulators, nnue->featureWeights[idx]);
        }
    } else {
        for (int i = 0; i < INNER_LAYER_COUNT; ++i)
            nnue->accumulators_perspective[i] = nnue->featureBias[i];
        for (int i = 0; i < NNUE_FT_INPUTS; ++i)
            nnue->inputs_perspective[i] = 0;
        for (int sq = 0; sq < 64; ++sq) {
            int pc = board->squares[sq];
            if (!pc) continue;
            int idx = getInputIndexOf(nnue->bkBucket, !pieceColor(pc), pieceType(pc), sq ^ PERSPECTIVE_MASK);
            nnue->inputs_perspective[idx] = 1;
            addFeature(nnue->accumulators_perspective, nnue->featureWeights[idx]);
        }
    }
}
#endif

// If a king changed bucket, that whole perspective is stale (every feature
// re-indexes) -> refresh it. No-op in the default (non-KB) build.
void nnueUpdateBuckets(Board* board) {
#ifdef NNUE_KB
    int wk = firstOne(board->colours[WHITE] & board->pieces[KING]);
    int bk = firstOne(board->colours[BLACK] & board->pieces[KING]);
    int nwk = nnueKingBucket(wk);
    int nbk = nnueKingBucket(bk ^ PERSPECTIVE_MASK);
    if (nwk != nnue->wkBucket) { nnue->wkBucket = nwk; refreshSide(board, WHITE); }
    if (nbk != nnue->bkBucket) { nnue->bkBucket = nbk; refreshSide(board, BLACK); }
#else
    (void) board;
#endif
}
