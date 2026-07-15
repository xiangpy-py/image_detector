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
///
/// # 返回
/// (训练集图像数, 测试集图像数)
pub fn preprocess_dataset_impl(
    root: &str,
    out: &str,
    size: u32,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let root = Path::new(root);
    let out = Path::new(out);

    let train_count = process_split(root, out, size, "train")?;
    let test_count = process_split(root, out, size, "test")?;

    Ok((train_count, test_count))
}

// ─── PyO3 Python 绑定 ───

#[pyfunction]
fn preprocess_dataset(root: &str, out: &str, size: u32) -> PyResult<(usize, usize)> {
    preprocess_dataset_impl(root, out, size)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn rust_preprocessor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(preprocess_dataset, m)?)?;
    Ok(())
}
