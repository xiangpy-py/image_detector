use std::path::Path;

use pyo3::prelude::*;

mod cache;
mod dataset;
mod image;
mod processor;

use processor::process_split;

/// 公共接口：处理数据集并生成 .npy 缓存。
///
/// # 参数
/// - `root`: 数据集根目录（包含 train/val/test 子目录）
/// - `out`: 缓存输出目录
/// - `size`: 图像目标尺寸（如 256）
/// - `merge_val`: 是否将 kaggle 数据集的 val 目录并入 train（默认 true）
///
/// # 返回
/// (训练集图像数, 测试集图像数)
pub fn preprocess_dataset_impl(
    root: &str,
    out: &str,
    size: u32,
    merge_val: bool,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let root = Path::new(root);
    let out = Path::new(out);

    let train_count = process_split(root, out, size, "train", merge_val)?;
    let test_count = process_split(root, out, size, "test", false)?;

    Ok((train_count, test_count))
}

// ─── PyO3 Python 绑定 ───

#[pyfunction]
#[pyo3(signature = (root, out, size, merge_val=true))]
fn preprocess_dataset(
    root: &str,
    out: &str,
    size: u32,
    merge_val: bool,
) -> PyResult<(usize, usize)> {
    preprocess_dataset_impl(root, out, size, merge_val)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn rust_preprocessor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(preprocess_dataset, m)?)?;
    Ok(())
}
