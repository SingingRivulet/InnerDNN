
// shader定义
static const char* shader_matmul =
    "#version 320 es\n"
    "uniform int n;\n"
    "uniform int x_offset;\n"
    "uniform int w_offset;\n"
    "layout(local_size_x = 1) in;\n"
    "layout(binding = 0) readonly buffer Input0{\n"
    "    float data[];\n"
    "} x;\n"

    "layout(binding = 1) readonly buffer Input1{\n"
    "    float data[];\n"
    "} w;\n"

    "layout(binding = 2) writeonly buffer Output0{\n"
    "    float data[];\n"
    "} xout;\n"

    "void main(){\n"
    "    int i = int(gl_GlobalInvocationID.x);\n"
    "    float val = 0.0;\n"
    "    for (int j = 0; j < n; j++) {\n"
    "        val += w.data[i * n + j + w_offset] * x.data[j + x_offset];\n"
    "    }\n"
    "    xout.data[i] = val;\n"
    "}\n";

static const char* shader_matmul_trans_vec4 =
    "#version 320 es\n"
    "uniform int n;\n"
    "uniform int d;\n"
    "uniform int x_offset;\n"
    "uniform int w_offset;\n"
    "layout(local_size_x = 1) in;\n"
    "layout(binding = 0) readonly buffer Input0{\n"
    "    float data[];\n"
    "} x;\n"

    "layout(binding = 1) readonly buffer Input1{\n"
    "    vec4 data[];\n"
    "} w;\n"

    "layout(binding = 2) writeonly buffer Output0{\n"
    "    vec4 data[];\n"
    "} xout;\n"

    "void main(){\n"
    "    int i = int(gl_GlobalInvocationID.x);\n"
    "    vec4 val = vec4(0.0 , 0.0 , 0.0 , 0.0);\n"
    "    for (int j = 0; j < d; j++) {\n"
    "        val += w.data[i  + j* n / 4 + w_offset / 4] * x.data[j + x_offset];\n"
    "    }\n"
    "    xout.data[i] = val;\n"
    "}\n";

static const char* shader_variance_before_sum =
    "#version 320 es\n"
    "uniform int insize;\n"
    "uniform int shape0;\n"
    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    float data[];\n"
    "} a;\n"

    "layout(binding = 1) writeonly buffer Output0{\n"
    "    float data[];\n"
    "} sum;\n"

    "layout(binding = 2) writeonly buffer Output1{\n"
    "    float data[];\n"
    "} b;\n"

    "void main(){\n"
    "    int idx = int(gl_GlobalInvocationID.x);\n"
    "    if(idx*2 >= insize){\n"
    "        b.data[idx] = 0.;\n"
    "        return;\n"
    "    }\n"
    "    float mean = sum.data[0]/float(insize);\n"
    "    float res = (a.data[idx*2]-mean)*(a.data[idx*2]-mean);\n"
    "    if(idx*2+1 < insize){\n"
    "       res += (a.data[idx*2+1]-mean)*(a.data[idx*2+1]-mean);\n"
    "    }\n"
    "    b.data[idx] = res;\n"
    "}\n";

static const char* shader_layerNorm_inplace =
    "#version 320 es\n"
    "uniform int insize;\n"
    "uniform int shape0;\n"
    "uniform int weight_offset;\n"
    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) buffer Input0{\n"
    "    float data[];\n"
    "} x;\n"

    "layout(binding = 1) readonly buffer Input1{\n"
    "    float data[];\n"
    "} sum;\n"

    "layout(binding = 2) readonly buffer Input2{\n"
    "    float data[];\n"
    "} vari_sum;\n"

    "layout(binding = 3) readonly buffer Input3{\n"
    "    float data[];\n"
    "} weight;\n"

    "layout(binding = 4) readonly buffer Input4{\n"
    "    float data[];\n"
    "} bias;\n"

    "void main(){\n"
    "    int idx = int(gl_GlobalInvocationID.x);\n"
    "    float mean = sum.data[0]/float(insize);\n"
    "    float vari = vari_sum.data[0]/float(insize);\n"
    "    x.data[idx] = (x.data[idx] - mean) / sqrt(vari + 0.00005) * weight.data[idx+weight_offset] + bias.data[idx+weight_offset];\n"
    "}\n";

static const char* shader_layerNorm =
    "#version 320 es\n"
    "uniform int insize;\n"
    "uniform int shape0;\n"
    "uniform int weight_offset;\n"
    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) buffer Input0{\n"
    "    float data[];\n"
    "} x;\n"

    "layout(binding = 1) readonly buffer Input1{\n"
    "    float data[];\n"
    "} sum;\n"

    "layout(binding = 2) readonly buffer Input2{\n"
    "    float data[];\n"
    "} vari_sum;\n"

    "layout(binding = 3) readonly buffer Input3{\n"
    "    float data[];\n"
    "} weight;\n"

    "layout(binding = 4) readonly buffer Input4{\n"
    "    float data[];\n"
    "} bias;\n"

    "layout(binding = 5) writeonly buffer Input5{\n"
    "    float data[];\n"
    "} o;\n"

    "void main(){\n"
    "    int idx = int(gl_GlobalInvocationID.x);\n"
    "    float mean = sum.data[0]/float(insize);\n"
    "    float vari = vari_sum.data[0]/float(insize);\n"
    "    o.data[idx] = (x.data[idx] - mean) / sqrt(vari + 0.00005) * weight.data[idx+weight_offset] + bias.data[idx+weight_offset];\n"
    "}\n";

static const char* shader_rmsnorm_squares_and_sum =
    "#version 320 es\n"
    "uniform int insize;\n"
    "uniform int shape0;\n"
    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    float data[];\n"
    "} a;\n"

    "layout(binding = 1) writeonly buffer Output0{\n"
    "    float data[];\n"
    "} b;\n"

    "void main(){\n"
    "    int idx = int(gl_GlobalInvocationID.x);\n"
    "    if(idx*2 >= insize){\n"
    "        b.data[idx] = 0.;\n"
    "        return;\n"
    "    }\n"
    "    float res = a.data[idx*2]*a.data[idx*2];\n"
    "    if(idx*2+1 < insize){\n"
    "       res += a.data[idx*2+1]*a.data[idx*2+1];\n"
    "    }\n"
    "    b.data[idx] = res;\n"
    "}\n";

static const char* shader_softmax_exp =
    "#version 320 es\n"
    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) buffer Input0{\n"
    "    float data[];\n"
    "} a;\n"

    "layout(binding = 1) readonly buffer Input1{\n"
    "    float data[];\n"
    "} maxVal_arr;\n"

    "void main(){\n"
    "    int idx = int(gl_GlobalInvocationID.x);\n"
    "    float max_val = maxVal_arr.data[0];\n"
    "    float res0 = exp(a.data[idx] - max_val);\n"
    "    a.data[idx] = res0;\n"
    "}\n";

static const char* shader_softmax_normalize =
    "#version 320 es\n"
    "uniform int shape0;\n"
    "layout(local_size_x = 1 , local_size_y = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    float data[];\n"
    "} sum_arr;\n"

    "layout(binding = 1) readonly buffer Input1{\n"
    "    float data[];\n"
    "} maxVal_arr;\n"

    "layout(binding = 2) buffer Input2{\n"
    "    float data[];\n"
    "} x;\n"

    "void main(){\n"
    "    float max_val = maxVal_arr.data[0];\n"
    "    int idx = int(gl_GlobalInvocationID.x);\n"
    "    int idy = int(gl_GlobalInvocationID.y);\n"
    "    x.data[idx + shape0*idy] = x.data[idx + shape0*idy]/sum_arr.data[idy];\n"
    "}\n";

static const char* shader_sum =
    "#version 320 es\n"
    "uniform int insize;\n"
    "uniform int shape0;\n"
    "layout(local_size_x = 1 , local_size_y = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    float data[];\n"
    "} a;\n"

    "layout(binding = 1) writeonly buffer Output0{\n"
    "    float data[];\n"
    "} b;\n"

    "void main(){\n"
    "    int idx = int(gl_GlobalInvocationID.x);\n"
    "    int idy = int(gl_GlobalInvocationID.y);\n"
    "    if(idx*2 >= insize){\n"
    "        b.data[idx + shape0*idy] = 0.;\n"
    "        return;\n"
    "    }\n"
    "    float res = a.data[insize*idy + idx*2];\n"
    "    if(idx*2+1 < insize){\n"
    "        res += a.data[insize*idy + idx*2 + 1];\n"
    "    }\n"
    "    b.data[idx + shape0*idy] = res;\n"
    "}\n";

static const char* shader_sum_vec4 =
    "#version 320 es\n"
    "uniform int insize;\n"
    "uniform int shape0;\n"
    "layout(local_size_x = 1 , local_size_y = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    vec4 data[];\n"
    "} a;\n"

    "layout(binding = 1) writeonly buffer Output0{\n"
    "    float data[];\n"
    "} b;\n"

    "void main(){\n"
    "    int idx = int(gl_GlobalInvocationID.x);\n"
    "    int idy = int(gl_GlobalInvocationID.y);\n"
    "    if(idx>=insize){\n"
    "        b.data[idx + shape0*idy] = 0.;\n"
    "        return;\n"
    "    }\n"
    "    vec4 va = a.data[insize*idy + idx];\n"

    "    float res0 = va.x + va.y;\n"  // step0-0
    "    float res1 = va.z + va.w;\n"  // step0-1

    "    b.data[idx + shape0*idy] = res0 + res1;\n"
    "}\n";

static const char* shader_max =
    "#version 320 es\n"
    "uniform int insize;\n"
    "uniform int shape0;\n"
    "layout(local_size_x = 1 , local_size_y = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    float data[];\n"
    "} a;\n"

    "layout(binding = 1) writeonly buffer Output0{\n"
    "    float data[];\n"
    "} b;\n"

    "const float infinity = 1. / 0.;\n"

    "void main(){\n"
    "    int idx = int(gl_GlobalInvocationID.x);\n"
    "    int idy = int(gl_GlobalInvocationID.y);\n"
    "    if(idx*2 >= insize){\n"
    "        b.data[idx + shape0*idy] = -infinity;\n"
    "        return;\n"
    "    }\n"
    "    if(idx*2+1 < insize){\n"
    "        b.data[idx + shape0*idy] = max(a.data[insize*idy + idx*2] , a.data[insize*idy + idx*2+1]);\n"
    "    }else{\n"
    "        b.data[idx + shape0*idy] = a.data[insize*idy + idx*2];\n"
    "    }\n"
    "}\n";

static const char* shader_max_vec4 =
    "#version 320 es\n"
    "uniform int insize;\n"
    "uniform int shape0;\n"
    "layout(local_size_x = 1 , local_size_y = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    vec4 data[];\n"
    "} a;\n"

    "layout(binding = 1) writeonly buffer Output0{\n"
    "    float data[];\n"
    "} b;\n"

    "const float infinity = 1. / 0.;\n"

    "void main(){\n"
    "    int idx = int(gl_GlobalInvocationID.x);\n"
    "    int idy = int(gl_GlobalInvocationID.y);\n"
    "    if(idx>=insize){\n"
    "        b.data[idx + shape0*idy] = -infinity;\n"
    "        return;\n"
    "    }\n"
    "    vec4 va = a.data[insize*idy + idx];\n"

    "    float res0 = max(va.x , va.y);\n"  // step0-0
    "    float res1 = max(va.z , va.w);\n"  // step0-1

    "    b.data[idx + shape0*idy] = max(res0 , res1);\n"
    "}\n";

static const char* shader_rmsnorm_normalize_and_scale =
    "#version 320 es\n"
    "uniform int size;\n"
    "uniform int weight_offset;\n"
    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    float data[];\n"
    "} ss_arr;\n"

    "layout(binding = 1) readonly buffer Input1{\n"
    "    float data[];\n"
    "} weight;\n"

    "layout(binding = 2) readonly buffer Input2{\n"
    "    float data[];\n"
    "} x;\n"

    "layout(binding = 3) writeonly buffer Output0{\n"
    "    float data[];\n"
    "} o;\n"

    "void main(){\n"
    "    int idx = int(gl_GlobalInvocationID.x);\n"
    "    float ss = ss_arr.data[0];\n"
    "    ss /= float(size);\n"
    "    ss += 0.00001;\n"
    "    ss = 1.0f / sqrt(ss);\n"
    "    o.data[idx] = weight.data[idx+weight_offset] * (ss * x.data[idx]);\n"
    "}\n";

static const char* shader_rmsnorm_normalize_and_scale_inplace =
    "#version 320 es\n"
    "uniform int size;\n"
    "uniform int weight_offset;\n"
    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    float data[];\n"
    "} ss_arr;\n"

    "layout(binding = 1) readonly buffer Input1{\n"
    "    float data[];\n"
    "} weight;\n"

    "layout(binding = 2) buffer Output0{\n"
    "    float data[];\n"
    "} o;\n"

    "void main(){\n"
    "    int idx = int(gl_GlobalInvocationID.x);\n"
    "    float ss = ss_arr.data[0];\n"
    "    ss /= float(size);\n"
    "    ss += 0.00001;\n"
    "    ss = 1.0f / sqrt(ss);\n"
    "    o.data[idx] = weight.data[idx+weight_offset] * (ss * o.data[idx]);\n"
    "}\n";

static const char* shader_accum =
    "#version 320 es\n"
    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) buffer Input0{\n"
    "    vec4 data[];\n"
    "} a;\n"

    "layout(binding = 1) readonly buffer Input1{\n"
    "    vec4 data[];\n"
    "} b;\n"

    "void main(){\n"
    "    int idx = int(gl_GlobalInvocationID.x);\n"
    "    a.data[idx] = a.data[idx]+b.data[idx];\n"
    "}\n";

static const char* shader_positionalEncoding =
    "#version 320 es\n"
    "uniform int pos;\n"
    "uniform int dim;\n"
    "uniform int hidden_dim;\n"
    "uniform int freq_cis_idx_delta;\n"
    "uniform int head_size;\n"

    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    vec2 data[];\n"
    "} freq_cis;\n"

    "layout(binding = 1) buffer Input1{\n"
    "    vec2 data[];\n"
    "} q;\n"

    "layout(binding = 2) buffer Input2{\n"
    "    vec2 data[];\n"
    "} k;\n"

    "void main(){\n"
    "    int idx = int(gl_GlobalInvocationID.x);\n"
    "    int i = idx*2;\n"
    "    vec2 qi = q.data[idx];\n"
    "    vec2 ki = k.data[idx];\n"
    "    vec2 fc = freq_cis.data[freq_cis_idx_delta+(i % head_size) / 2];\n"
    "    q.data[idx] = vec2(qi.x * fc.x - qi.y * fc.y , qi.x * fc.y + qi.y * fc.x);\n"
    "    k.data[idx] = vec2(ki.x * fc.x - ki.y * fc.y , ki.x * fc.y + ki.y * fc.x);\n"
    "}\n";

static const char* shader_transformer_silu_and_mulW3 =
    "#version 320 es\n"
    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) buffer Input0{\n"
    "    vec4 data[];\n"
    "} hb;\n"

    "layout(binding = 1) readonly buffer Input1{\n"
    "    vec4 data[];\n"
    "} hb2;\n"

    "void main(){\n"
    "    int idx = int(gl_GlobalInvocationID.x);\n"
    "    vec4 v = hb.data[idx];\n"
    "    vec4 res = v * (vec4(1.,1.,1.,1.) / (vec4(1.,1.,1.,1.) + exp(-v))) * hb2.data[idx];\n"
    "    hb.data[idx] = res;\n"
    "}\n";

static const char* shader_transformer_get_query_vector =
    "#version 320 es\n"
    "uniform int seq_len;\n"
    "uniform int pos;\n"
    "uniform int head_size;\n"
    "uniform int dim;\n"
    "uniform int layer_idx;\n"

    "layout(local_size_x = 1 , local_size_y = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    float data[];\n"
    "} q;\n"

    "layout(binding = 1) readonly buffer Input1{\n"
    "    float data[];\n"
    "} k;\n"

    "layout(binding = 2) writeonly buffer Output0{\n"
    "    float data[];\n"
    "} att;\n"

    "void main(){\n"
    "    int h = int(gl_GlobalInvocationID.x);\n"
    "    int t = int(gl_GlobalInvocationID.y);\n"
    "    int loff = layer_idx * seq_len * dim;\n"
    "    int q_offset = h * head_size;\n"
    "    int att_offset = h * seq_len;\n"
    "    int k_offset = loff + t * dim + h * head_size;\n"
    "    float score = 0.0;\n"
    "    for (int i = 0; i < head_size; i++) {\n"
    "        score += q.data[i+q_offset] * k.data[i+k_offset];\n"
    "    }\n"
    "    score /= sqrt(float(head_size));\n"
    "    att.data[t+att_offset] = score;\n"
    "}\n";

static const char* shader_transformer_build_attMat =
    "#version 320 es\n"
    "uniform int seq_len;\n"
    "uniform int pos;\n"
    "uniform int head_size;\n"
    "uniform int dim;\n"
    "uniform int layer_idx;\n"

    "layout(local_size_x = 1 , local_size_y = 1, local_size_z = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    float data[];\n"
    "} value_cache;\n"

    "layout(binding = 1) readonly buffer Input1{\n"
    "    float data[];\n"
    "} att;\n"

    "layout(binding = 2) writeonly buffer Output0{\n"
    "    float data[];\n"
    "} attMat;\n"

    "void main(){\n"
    "    int h = int(gl_GlobalInvocationID.x);\n"
    "    int i = int(gl_GlobalInvocationID.y);\n"
    "    int t = int(gl_GlobalInvocationID.z);\n"
    "    int xb_offset = h * head_size;\n"
    "    int loff = layer_idx * seq_len * dim;\n"
    "    int att_offset = h * seq_len;\n"
    "    int v_offset = loff + t * dim + h * head_size;\n"
    "    float a = att.data[t+att_offset];\n"
    "    float attMatVal = a * value_cache.data[i+v_offset];\n"
    "    attMat.data[h*(pos+1)*head_size + i*(pos+1) + t] = attMatVal;\n"
    "}\n";

static const char* shader_transformer_softmax_input =
    "#version 320 es\n"
    "uniform int seq_len;\n"
    "uniform int pos;\n"

    "layout(local_size_x = 1, local_size_y = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    float data[];\n"
    "} src;\n"

    "layout(binding = 1) writeonly buffer Output0{\n"
    "    float data[];\n"
    "} dst;\n"

    "void main(){\n"
    "    int h = int(gl_GlobalInvocationID.x);\n"
    "    int t = int(gl_GlobalInvocationID.y);\n"
    "    int srcIdx = h * seq_len + t;\n"
    "    int dstIdx = h * (pos+1) + t;\n"
    "    dst.data[dstIdx] = src.data[srcIdx];\n"
    "}\n";

static const char* shader_transformer_softmax_output =
    "#version 320 es\n"
    "uniform int seq_len;\n"
    "uniform int pos;\n"

    "layout(local_size_x = 1, local_size_y = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    float data[];\n"
    "} src;\n"

    "layout(binding = 1) writeonly buffer Output0{\n"
    "    float data[];\n"
    "} dst;\n"

    "void main(){\n"
    "    int h = int(gl_GlobalInvocationID.x);\n"
    "    int t = int(gl_GlobalInvocationID.y);\n"
    "    int dstIdx = h * seq_len + t;\n"
    "    int srcIdx = h * (pos+1) + t;\n"
    "    dst.data[dstIdx] = src.data[srcIdx];\n"
    "}\n";

static const char* shader_temperature =
    "#version 320 es\n"
    "uniform float temperature;\n"

    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) buffer Input0{\n"
    "    float data[];\n"
    "} logit;\n"

    "void main(){\n"
    "    int index = int(gl_GlobalInvocationID.x);\n"
    "    logit.data[index] /= temperature;\n"
    "}\n";

static const char* shader_copyBuffer =
    "#version 320 es\n"
    "uniform int src_offset;\n"
    "uniform int dst_offset;\n"

    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    float data[];\n"
    "} src;\n"

    "layout(binding = 1) writeonly buffer Output0{\n"
    "    float data[];\n"
    "} dst;\n"

    "void main(){\n"
    "    int index = int(gl_GlobalInvocationID.x);\n"
    "    dst.data[index+dst_offset] = src.data[index+src_offset];\n"
    "}\n";

static const char* shader_rwkv_att_wkv =
    "#version 320 es\n"

    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    vec4 data[];\n"
    "} att_time_first;\n"

    "layout(binding = 1) readonly buffer Input1{\n"
    "    vec4 data[];\n"
    "} att_time_decay;\n"

    "layout(binding = 2) readonly buffer Input2{\n"
    "    vec4 data[];\n"
    "} k;\n"

    "layout(binding = 3) readonly buffer Input3{\n"
    "    vec4 data[];\n"
    "} v;\n"

    "layout(binding = 4) buffer Input4{\n"
    "    vec4 data[];\n"
    "} aa;\n"

    "layout(binding = 5) buffer Input5{\n"
    "    vec4 data[];\n"
    "} bb;\n"

    "layout(binding = 6) buffer Input6{\n"
    "    vec4 data[];\n"
    "} pp;\n"

    "layout(binding = 7) writeonly buffer Output0{\n"
    "    vec4 data[];\n"
    "} wkv;\n"

    "void main(){\n"
    "    int index = int(gl_GlobalInvocationID.x);\n"
    "    vec4 pp_v = pp.data[index];\n"
    "    vec4 k_v = k.data[index];\n"
    "    vec4 v_v = v.data[index];\n"
    "    vec4 ww = att_time_first.data[index] + k_v;\n"
    "    vec4 qq = max(ww , pp_v);\n"
    "    vec4 e1 = exp(pp_v - qq);\n"
    "    vec4 e2 = exp(ww   - qq);\n"
    "    vec4 a = e1 * aa.data[index] + e2 * v_v;\n"
    "    vec4 b = e1 * bb.data[index] + e2;\n"
    "    ww = pp.data[index] + time_decay.data[index];\n"
    "    qq = max(ww, k_v);\n"
    "    e1 = exp(ww - qq);\n"
    "    e2 = exp(k_v - qq);\n"
    "    aa.data[index] = e1 * aa.data[index] + e2 * v_v;\n"
    "    bb.data[index] = e1 * bb.data[index] + e2;\n"
    "    pp.data[index] = qq;\n"
    "    wkv.data[index]= a/b;\n"
    "}\n";

static const char* shader_rwkv_att_rkv =
    "#version 320 es\n"

    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    vec4 data[];\n"
    "} att_time_mix_k;\n"

    "layout(binding = 1) readonly buffer Input1{\n"
    "    vec4 data[];\n"
    "} att_time_mix_v;\n"

    "layout(binding = 2) readonly buffer Input2{\n"
    "    vec4 data[];\n"
    "} att_time_mix_r;\n"

    "layout(binding = 3) readonly buffer Input3{\n"
    "    vec4 data[];\n"
    "} x;\n"

    "layout(binding = 4) readonly buffer Input4{\n"
    "    vec4 data[];\n"
    "} x_prev;\n"

    "layout(binding = 5) writeonly buffer Input5{\n"
    "    vec4 data[];\n"
    "} xr;\n"

    "layout(binding = 6) writeonly buffer Input6{\n"
    "    vec4 data[];\n"
    "} xk;\n"

    "layout(binding = 7) writeonly buffer Output0{\n"
    "    vec4 data[];\n"
    "} xv;\n"

    "void main(){\n"
    "    int index = int(gl_GlobalInvocationID.x);\n"
    "    vec4 x_v = x.data[index];\n"
    "    vec4 x_prev_v = x_prev.data[index];\n"
    "    vec4 time_mix_k_v = time_mix_k.data[index];\n"
    "    vec4 time_mix_v_v = time_mix_v.data[index];\n"
    "    vec4 time_mix_r_v = time_mix_r.data[index];\n"
    "    xk.data[index] = x_v * time_mix_k_v + x_prev_v * (vec4(1.)-time_mix_k_v);\n"
    "    xv.data[index] = x_v * time_mix_v_v + x_prev_v * (vec4(1.)-time_mix_v_v);\n"
    "    xr.data[index] = x_v * time_mix_r_v + x_prev_v * (vec4(1.)-time_mix_r_v);\n"
    "}\n";

static const char* shader_rwkv_ffn =
    "#version 320 es\n"

    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    vec4 data[];\n"
    "} att_time_mix_k;\n"

    "layout(binding = 1) readonly buffer Input1{\n"
    "    vec4 data[];\n"
    "} att_time_mix_r;\n"

    "layout(binding = 2) readonly buffer Input2{\n"
    "    vec4 data[];\n"
    "} x;\n"

    "layout(binding = 3) readonly buffer Input3{\n"
    "    vec4 data[];\n"
    "} x_prev;\n"

    "layout(binding = 4) writeonly buffer Input4{\n"
    "    vec4 data[];\n"
    "} xr;\n"

    "layout(binding = 5) writeonly buffer Input5{\n"
    "    vec4 data[];\n"
    "} xk;\n"

    "void main(){\n"
    "    int index = int(gl_GlobalInvocationID.x);\n"
    "    vec4 x_v = x.data[index];\n"
    "    vec4 x_prev_v = x_prev.data[index];\n"
    "    vec4 time_mix_k_v = time_mix_k.data[index];\n"
    "    vec4 time_mix_r_v = time_mix_r.data[index];\n"
    "    xk.data[index] = x_v * time_mix_k_v + x_prev_v * (vec4(1.) - time_mix_k_v);\n"
    "    xr.data[index] = x_v * time_mix_r_v + x_prev_v * (vec4(1.) - time_mix_r_v);\n"
    "}";

static const char* shader_sigmoid =
    "#version 320 es\n"

    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    vec4 data[];\n"
    "} x;\n"

    "layout(binding = 1) writeonly buffer Output0{\n"
    "    vec4 data[];\n"
    "} xout;\n"

    "void main(){\n"
    "    int index = int(gl_GlobalInvocationID.x);\n"
    "    xout.data[index] = vec4(1.) / (vec4(1.) - exp(-x.data[index]));\n"
    "}";

static const char* shader_vecxvec =
    "#version 320 es\n"

    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    vec4 data[];\n"
    "} a;\n"

    "layout(binding = 1) readonly buffer Input1{\n"
    "    vec4 data[];\n"
    "} b;\n"

    "layout(binding = 2) writeonly buffer Output0{\n"
    "    vec4 data[];\n"
    "} out;\n"

    "void main(){\n"
    "    int index = int(gl_GlobalInvocationID.x);\n"
    "    out.data[index] = a.data[index] * b.data[index];\n"
    "}";

static const char* shader_reluAndsqr =
    "#version 320 es\n"

    "layout(local_size_x = 1) in;\n"

    "layout(binding = 0) readonly buffer Input0{\n"
    "    vec4 data[];\n"
    "} x;\n"

    "layout(binding = 1) writeonly buffer Output0{\n"
    "    vec4 data[];\n"
    "} xout;\n"

    "vec4 sqr(vec4 x){\n"
    "    return x*x;\n"
    "}\n"

    "vec4 relu(vec4 x){\n"
    "    return max(vec4(0.) , x);\n"
    "}\n"

    "void main(){\n"
    "    int index = int(gl_GlobalInvocationID.x);\n"
    "    xout.data[index] = sqr(relu(x.data[index]));\n"
    "}";
