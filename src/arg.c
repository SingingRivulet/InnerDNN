#include "innerDNN/arg.h"

long innerDNN_time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

static unsigned long long rng_seed;
unsigned int innerDNN_random_u32() {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    rng_seed ^= rng_seed >> 12;
    rng_seed ^= rng_seed << 25;
    rng_seed ^= rng_seed >> 27;
    return (rng_seed * 0x2545F4914F6CDD1Dull) >> 32;
}
float innerDNN_random_f32() {  // random float32 in [0,1)
    return (innerDNN_random_u32() >> 8) / 16777216.0f;
}

// ----------------------------------------------------------------------------
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

int innerDNN_argmax(innerDNN_shader_programs* prog, GLuint probabilities_gpu, int n) {
    // return the index that has the highest probability
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, probabilities_gpu);
    float* probabilities = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                                    n * sizeof(float), GL_MAP_READ_BIT);
    innerDNN_GPU_CHECK();
    int max_i = 0;
    float max_p;
    if (probabilities) {
        max_p = probabilities[0];
        for (int i = 1; i < n; i++) {
            if (probabilities[i] > max_p) {
                max_i = i;
                max_p = probabilities[i];
            }
        }
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    innerDNN_GPU_CHECK();

    // printf("max_i=%d,max_p=%f\n", max_i, max_p);
    return max_i;
}

int innerDNN_sample(innerDNN_shader_programs* prog, GLuint probabilities_gpu, int n) {
    // sample index from probabilities (they must sum to 1!)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, probabilities_gpu);
    float* probabilities = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                                    n * sizeof(float), GL_MAP_READ_BIT);
    innerDNN_GPU_CHECK();
    int res = n - 1;
    if (probabilities) {
        float r = innerDNN_random_f32();
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probabilities[i];
            if (r < cdf) {
                res = i;
                break;
            }
        }
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    return res;
}

int innerDNN_compare(const void* a, const void* b) {
    innerDNN_probIndex* a_ = (innerDNN_probIndex*)a;
    innerDNN_probIndex* b_ = (innerDNN_probIndex*)b;
    if (a_->prob > b_->prob)
        return -1;
    if (a_->prob < b_->prob)
        return 1;
    return 0;
}

int innerDNN_sample_topp(innerDNN_shader_programs* prog, GLuint probabilities_gpu, int n, float topp, innerDNN_probIndex* probIndex) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".

    // quicksort indices in descending order of probabilities
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, probabilities_gpu);
    float* probabilities = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                                    n * sizeof(float), GL_MAP_READ_BIT);
    innerDNN_GPU_CHECK();
    int res = n - 1;
    if (probabilities) {
        for (int i = 0; i < n; i++) {
            probIndex[i].index = i;
            probIndex[i].prob = probabilities[i];
        }
        qsort(probIndex, n, sizeof(innerDNN_probIndex), innerDNN_compare);

        // truncate the list where cumulative probability exceeds topp
        float cumulative_prob = 0.0f;
        int last_idx = 0;
        for (int i = 0; i < n; i++) {
            cumulative_prob += probIndex[i].prob;
            if (cumulative_prob > topp) {
                last_idx = i;
                break;  // we've exceeded topp by including last_idx
            }
        }

        // sample from the truncated list
        float r = innerDNN_random_f32() * cumulative_prob;
        float cdf = 0.0f;
        for (int i = 0; i <= last_idx; i++) {
            cdf += probIndex[i].prob;
            if (r < cdf) {
                res = probIndex[i].index;
                break;
            }
        }
        res = probIndex[last_idx].index;  // in case of rounding errors
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    return res;
}