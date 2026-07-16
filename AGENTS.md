# Agent 操作指南

## 一、环境

- 包管理器：`uv`
- 运行命令：`uv run python ...`
- 环境同步：`uv sync`
- 虚拟环境：`.venv/`（勿手动操作）

## 二、面对新项目的第一步

**先理解，再行动**：

1. **看目录结构** → 了解项目规模和模块划分
2. **看入口文件** → 了解程序启动方式和命令路由
3. **看配置文件** → 了解可配置项和默认值
4. **看依赖清单** → 了解技术栈（pyproject.toml、requirements.txt）
5. **看文档** → AGENTS.md、README.md

**禁止**：在不了解项目结构的情况下直接开始改代码

## 三、任务拆分方法

**按层次拆分**：
```
任务 → 模块 → 函数/类 → 文件
```

**优先级判断**：

| 信号 | 优先级 |
|------|--------|
| 用户明确要求 | 高 |
| 阻塞其他任务 | 高 |
| 简单独立任务 | 先做（快速完成） |
| 涉及多模块改动 | 后做（先规划） |
| 需要重构架构 | 最后（谨慎） |

**拆分粒度**：
- 一个任务 = 一个清晰的变更目标
- 一个子任务 = 一个文件/模块内的改动
- 每个子任务完成后立即验证，不要堆到最后

## 四、文件夹分配原则

**按职责分层**：

```
项目根目录/
├── src/           # 核心源代码（业务逻辑）
├── tests/         # 测试代码
├── docs/          # 文档
├── configs/       # 配置文件（可选）
├── scripts/       # 脚本工具（部署、运维）
├── data/          # 数据文件（可选）
├── outputs/       # 输出结果（可被清理）
├── cache/         # 缓存（可被清理）
└── models/        # 模型文件（机器学习项目）
```

**命名约定**：
- 源码目录：`src`、`src-py`、`lib`、`app`
- 测试目录：`tests`、`test`
- 入口文件：`main.py`、`app.py`

## 五、代码修改原则

**最小改动原则**：

1. 只改必要的代码，不要顺手"优化"无关部分
2. 不添加用户未要求的注释
3. 不重构未涉及的模块

**修改前必须先读**：
- 读相关文件了解现有逻辑
- 读配置了解可配置项
- 读测试了解边界情况

**修改后必须验证**：
- 运行相关测试
- 运行类型检查（如有）
- 运行代码检查（lint）

## 六、配置管理原则

**三层配置优先级**：
```
环境变量 > 命令行参数 > 默认值
```

**路径配置**：
- 默认值放在项目根目录（可移植）
- 允许环境变量覆盖（CI/CD 场景、实时修改）
- 允许命令行覆盖（临时场景）

**环境变量实时修改路径**：
```bash
# 临时修改（当前会话有效）
export MODELS_DIR=/path/to/models
export OUTPUTS_DIR=/path/to/outputs
uv run python main.py train

# 或在命令前直接设置（仅本次有效）
MODELS_DIR=/path/to/models uv run python main.py train
```

**不要硬编码路径**：
```python
# 错误
path = "/home/user/data"

# 正确
path = Path.home() / "data"
# 或
path = PROJECT_ROOT / "data"
```

## 七、任务执行流程

```
1. 理解需求 → 问清楚边界条件
2. 探索代码 → 读相关文件
3. 制定方案 → 确认改动范围
4. 执行改动 → 小步快跑
5. 验证结果 → 运行测试/命令
6. 汇报完成 → 简洁总结
```

**遇到阻塞时**：
- 不要反复尝试同一方法
- 换个角度思考
- 问用户澄清需求
- 搜索相关文档/社区方案
- 网络搜索同样类型项目做参考

**网络搜索参考**：
- 搜索关键词：项目类型 + "best practices" / "github" / "example"
- 例如：`deep learning image classification best practices github`
- 参考开源项目的目录结构、配置方式、代码风格
- 不要直接复制代码，理解原理后适配到本项目

## 八、本项目特定规范

### 项目结构
```
项目根目录/
├── main.py              # 入口文件（仅做路由）
├── src-py/              # Python 源代码
│   ├── config.py        # 全局配置（超参数、常量）
│   ├── system.py        # 系统配置（路径、多进程）
│   ├── dataset.py       # 数据加载
│   ├── train.py         # 训练逻辑
│   ├── evaluate.py      # 评估逻辑
│   ├── model.py         # 模型定义
│   ├── metrics.py       # 指标计算
│   └── gui.py           # GUI 模块
├── src/                 # Rust 预处理扩展
├── cache/               # 图像缓存（默认）
├── models/              # 模型文件（默认）
├── outputs/             # 输出文件（默认）
└── datasets.json        # 数据集注册表（默认）
```

### 环境变量配置示例
```bash
export DATASET_ROOT=/path/to/dataset
export DATASET_ROOTS=/path/to/dataset1,/path/to/dataset2
export CACHE_DIR=/path/to/cache
export MODELS_DIR=/path/to/models
export OUTPUTS_DIR=/path/to/outputs
```

### 数据处理流程
```bash
# 1. 下载数据集（可选）
uv run python main.py download

# 2. 注册数据集（可选）
uv run python main.py dataset add chest1 /path/to/data

# 3. 生成缓存（必需）
uv run python main.py cache

# 4. 训练模型
uv run python main.py train

# 5. 评估模型
uv run python main.py evaluate
```

### 关键约定
- **缓存格式**：`(N, 3, H, W)` uint8 numpy 数组
- **图像尺寸**：缓存 256，训练输入 224
- **训练保存**：每个 epoch 保存历史，防止中断丢失
- **早停机制**：patience=10（`EARLY_STOP_PATIENCE`），scheduler patience=5（`SCHEDULER_PATIENCE`）
- **日志**：使用 `loguru`，格式 `logger.info(...)`