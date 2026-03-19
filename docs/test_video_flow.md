# DCVC-RT 测试代码流程详解

## 1. 入口脚本

**文件**: `test_video.sh`

```bash
python test_video.py \
    --model_path_i ./checkpoints/cvpr2025_image.pth.tar \
    --model_path_p ./checkpoints/cvpr2025_video.pth.tar \
    --rate_num 8 \
    --test_config ./dataset_config_example_rgb_classB.json \
    --cuda 1 -w 1 --write_stream 1 \
    --force_zero_thres 0.12 \
    --output_path output.json \
    --force_intra_period -1 \
    --reset_interval 64 \
    --force_frame_num -1
```

---

## 2. 主函数流程 (`test_video.py`)

### 2.1 `main()` — 主入口

**文件**: `test_video.py:417-537`

**流程**:
1. `parse_args()` — 解析命令行参数
2. 加载 `test_config` JSON 配置
3. 计算 QP 码率点列表 (根据 `rate_num`)
4. 创建进程池 `ProcessPoolExecutor`
5. 遍历所有(数据集, 序列, 码率点)组合, 提交 `worker` 任务
6. 收集结果, 写入 `output.json`

### 2.2 `init_func()` — Worker进程模型初始化

**文件**: `test_video.py:381-414`

| 步骤 | 函数/类 | 文件 |
|------|---------|------|
| 初始化GPU | `set_torch_env()` | `src/utils/common.py` |
| 创建I帧模型 | `DMCI()` | `src/models/image_model.py` |
| 加载权重 | `get_state_dict()`, `load_state_dict()` | `src/utils/common.py` |
| 模型预处理 | `update()`, `half()` | DMCI类方法 |
| 创建P帧模型 | `DMC()` | `src/models/video_model.py` |
| 加载权重 | 同上 | 同上 |
| 模型预处理 | `update()`, `half()` | DMC类方法 |

### 2.3 `worker()` — 单序列测试任务

**文件**: `test_video.py:354-378`

| 步骤 | 函数/变量 | 说明 |
|------|-----------|------|
| 构建路径 | `args['src_path']` | `{dataset_path}/{seq}` |
| 码流路径 | `args['curr_bin_path']` | `out_bin/{ds_name}/{seq}_q{qp}.bin` |
| JSON路径 | `args['curr_json_path']` | 同上, `.bin` → `.json` |
| 调用测试 | `run_one_point_with_stream()` | 核心编码解码流程 |

---

## 3. 核心测试流程 (`run_one_point_with_stream()`)

**文件**: `test_video.py:130-347`

### Phase 1: 编码 + 写码流

| 步骤 | 函数/操作 | 文件/类 |
|------|-----------|---------|
| 创建源读取器 | `get_src_reader()` → `PNGReader` / `YUV420Reader` | `src/utils/video_reader.py` |
| 获取padding | `DMCI.get_padding_size()` | `src/models/image_model.py:102` |
| 设置熵编码器数量 | `set_use_two_entropy_coders()` | DMC/DMCI类 |
| 重置POC | `p_frame_net.set_curr_poc(0)` | `src/models/video_model.py:271` |

#### 帧级编码循环 (`for frame_idx in range(frame_num)`):

**读取帧**:
| 步骤 | 函数 | 文件 |
|------|------|------|
| 读取原始帧 | `src_reader.read_one_frame()` | `src/utils/video_reader.py` |
| YUV420→YUV444 | `ycbcr420_to_444_np()` | `src/utils/transforms.py` |
| PNG→Tensor | `np_image_to_tensor()` | `test_video.py:60` |
| RGB→YCbCr | `rgb2ycbcr()` | `src/utils/transforms.py` |
| Padding | `replicate_pad()` | `src/layers/cuda_inference.py` |

**I帧编码** (条件: `frame_idx == 0` 或 `frame_idx % intra_period == 0`):
| 步骤 | 函数/操作 | 文件 |
|------|-----------|------|
| I帧压缩 | `i_frame_net.compress(x_padded, qp_i)` → `{'x_hat', 'bit_stream'}` | `src/models/image_model.py:143` |
| 清空DPB | `p_frame_net.clear_dpb()` | `src/models/video_model.py:268` |
| 添加参考帧 | `p_frame_net.add_ref_frame(None, encoded['x_hat'])` | `src/models/video_model.py:257` |

**P帧编码**:
| 步骤 | 函数/操作 | 文件 |
|------|-----------|------|
| 特征适配器 | `p_frame_net.prepare_feature_adaptor_i(last_qp)` (条件满足时) | DMC类 |
| QP偏移 | `p_frame_net.shift_qp(qp_p, fa_idx)` | `src/models/video_model.py:378` |
| P帧压缩 | `p_frame_net.compress(x_padded, curr_qp)` → `{'x_hat', 'bit_stream'}` | `src/models/video_model.py:299` |

**写码流**:
| 步骤 | 函数 | 文件 |
|------|------|------|
| 获取SPS ID | `sps_helper.get_sps_id(sps)` | `src/utils/stream_helper.py:114` |
| 写SPS | `write_sps(output_buff, sps)` | `src/utils/stream_helper.py:148` |
| 写IP帧 | `write_ip(output_buff, is_i_frame, sps_id, qp, encoded['bit_stream'])` | `src/utils/stream_helper.py:198` |

### Phase 2: 解码 + 计算质量

| 步骤 | 函数/操作 | 文件/类 |
|------|-----------|---------|
| 重新打开码流 | `open(args['curr_bin_path'], "rb")` | Python内置 |
| 重新创建源读取器 | `get_src_reader()` | 同上 |
| 创建重建帧写入器 | `PNGWriter` / `YUV420Writer` (可选) | `src/utils/video_writer.py` |
| 重置POC | `p_frame_net.set_curr_poc(0)` | DMC类 |

#### 帧级解码循环 (`while decoded_frame_number < frame_num`):

| 步骤 | 函数 | 文件 |
|------|------|------|
| 读header | `read_header(input_buff)` | `src/utils/stream_helper.py:165` |
| 读SPS | `read_sps_remaining(input_buff, sps_id)` | `src/utils/stream_helper.py:187` |
| 添加SPS | `sps_helper.add_sps_by_id(sps)` | `src/utils/stream_helper.py:114` |
| 读QP和码流 | `read_ip_remaining(input_buff)` | `src/utils/stream_helper.py:212` |
| I帧解码 | `i_frame_net.decompress(bit_stream, sps, qp)` | `src/models/image_model.py:187` |
| P帧解码 | `p_frame_net.decompress(bit_stream, sps, qp)` | `src/models/video_model.py:343` |
| 更新参考帧 | `add_ref_frame()` / `clear_dpb()` | DMC类 |

**计算失真**:
| 步骤 | 函数 | 文件 |
|------|------|------|
| YUV444→YUV420 | `yuv_444_to_420()` | `src/utils/transforms.py` |
| YCbCr→RGB | `ycbcr2rgb()` | `src/utils/transforms.py` |
| 计算PSNR | `calc_psnr()` | `src/utils/metrics.py` |
| 计算SSIM | `calc_msssim()` / `calc_msssim_rgb()` | `src/utils/metrics.py` |

**写结果**:
| 步骤 | 函数 | 文件 |
|------|------|------|
| 生成日志JSON | `generate_log_json()` | `src/utils/common.py` |
| 保存JSON | `json.dump()` | Python内置 |

---

## 4. 模型类

### 4.1 DMCI (I帧模型)

**文件**: `src/models/image_model.py:102`

| 方法 | 行号 | 功能 |
|------|------|------|
| `compress(x, qp)` | 143 | I帧编码, 返回 `{'x_hat', 'bit_stream'}` |
| `decompress(bit_stream, sps, qp)` | 187 | I帧解码, 返回 `{'x_hat'}` |
| `get_padding_size(h, w, divisor)` | 102 | 获取padding尺寸 |
| `update()` | 父类 | 更新量化阈值 |
| `half()` | PyTorch | 转FP16 |

### 4.2 DMC (P帧模型)

**文件**: `src/models/video_model.py:226`

| 方法 | 行号 | 功能 |
|------|------|------|
| `compress(x, qp)` | 299 | P帧编码, 返回 `{'x_hat', 'bit_stream'}` |
| `decompress(bit_stream, sps, qp)` | 343 | P帧解码, 返回 `{'x_hat'}` |
| `add_ref_frame(feature, frame, increase_poc)` | 257 | 添加参考帧到DPB |
| `clear_dpb()` | 268 | 清空参考帧缓冲 |
| `set_curr_poc(poc)` | 271 | 设置当前POC |
| `shift_qp(qp, fa_idx)` | 378 | 根据帧索引偏移QP |
| `prepare_feature_adaptor_i(qp)` | DMC类 | 准备特征适配器 |
| `reset_ref_feature()` | DMC类 | 重置参考特征 |
| `set_use_two_entropy_coders()` | 父类 | 设置熵编码器数量 |
| `update()` | 父类 | 更新量化阈值 |

---

## 5. 工具类/函数

### 5.1 码流处理 (`src/utils/stream_helper.py`)

| 类/函数 | 行号 | 功能 |
|---------|------|------|
| `NalType` (enum) | 108 | NAL类型: `NAL_SPS`, `NAL_I`, `NAL_P` |
| `SPSHelper` | 114 | SPS管理: `get_sps_id()`, `add_sps_by_id()`, `get_sps_by_id()` |
| `write_sps(f, sps)` | 148 | 写SPS到码流 |
| `write_ip(f, is_i_frame, sps_id, qp, bit_stream)` | 198 | 写I/P帧头和码流 |
| `read_header(f)` | 165 | 读NAL header |
| `read_sps_remaining(f, sps_id)` | 187 | 读SPS剩余数据 |
| `read_ip_remaining(f)` | 212 | 读I/P帧剩余数据 |

### 5.2 视频读写 (`src/utils/video_reader.py`, `src/utils/video_writer.py`)

| 类 | 文件 | 功能 |
|----|------|------|
| `PNGReader` | `video_reader.py` | 读取PNG序列 |
| `YUV420Reader` | `video_reader.py` | 读取YUV420文件 |
| `PNGWriter` | `video_writer.py` | 写入PNG序列 |
| `YUV420Writer` | `video_writer.py` | 写入YUV420文件 |

### 5.3 指标计算 (`src/utils/metrics.py`)

| 函数 | 功能 |
|------|------|
| `calc_psnr()` | 计算PSNR |
| `calc_msssim()` | 计算YUV的MS-SSIM |
| `calc_msssim_rgb()` | 计算RGB的MS-SSIM |

### 5.4 颜色空间转换 (`src/utils/transforms.py`)

| 函数 | 功能 |
|------|------|
| `rgb2ycbcr()` | RGB → YCbCr |
| `ycbcr2rgb()` | YCbCr → RGB |
| `yuv_444_to_420()` | YUV444 → YUV420 |
| `ycbcr420_to_444_np()` | YUV420 → YUV444 |

---

## 6. 输出文件

| 文件 | 路径 | 内容 |
|------|------|------|
| 码流 | `out_bin/{ds_name}/{seq}_q{qp}.bin` | 编码后的二进制码流 |
| 重建帧 | 可选: `{seq}_{kbps}kbps.yuv` 或 PNG目录 | 解码重建的视频 |
| JSON | `out_bin/{ds_name}/{seq}_q{qp}.json` | 详细测试结果 |
| 汇总JSON | `output_path` (命令行指定) | 所有序列的汇总结果 |

---

## 7. 流程图

```
main()
├── parse_args()
├── load test_config.json
├── 计算QP列表
├── ProcessPoolExecutor(max_workers=w)
│   └── worker() [每个序列+码率点一个任务]
│       ├── init_func() [每个进程调用一次]
│       │   ├── DMCI() + load_state_dict()  # I帧模型
│       │   └── DMC() + load_state_dict()   # P帧模型
│       └── run_one_point_with_stream()
│           ├── Phase 1: 编码
│           │   ├── for frame_idx in range(frame_num):
│           │   │   ├── get_src_frame() → 读取原始帧
│           │   │   ├── replicate_pad() → padding
│           │   │   ├── [I帧] i_frame_net.compress()
│           │   │   └── [P帧] p_frame_net.compress()
│           │   └── write_sps() + write_ip() → .bin
│           │
│           └── Phase 2: 解码
│               ├── while decoded < frame_num:
│               │   ├── read_header() + read_ip_remaining()
│               │   ├── [I帧] i_frame_net.decompress()
│               │   ├── [P帧] p_frame_net.decompress()
│               │   ├── get_distortion() → PSNR/SSIM
│               │   └── [可选] PNGWriter/YUV420Writer
│               └── generate_log_json() → .json
│
└── dump_json(output.json)
```
