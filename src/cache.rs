use std::fs;
use std::path::Path;

use ndarray::{s, Array3, Array4};
use ndarray_npy::write_npy;

/// 将一个 split 的处理结果写入 .npy 缓存。
///
/// # 参数
/// - `out`: 缓存输出目录
/// - `split_name`: split 名称，如 "train" / "test"
/// - `size`: 图像尺寸
/// - `images`: 图像数组列表，每张形状为 (3, size, size)
/// - `labels`: 标签列表
pub fn save_split_cache(
    out: &Path,
    split_name: &str,
    size: u32,
    images: &[Array3<u8>],
    labels: &[i64],
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out)?;

    let n = images.len();
    let sz = size as usize;
    let mut image_array = Array4::<u8>::zeros((n, 3, sz, sz));
    let mut label_array = ndarray::Array1::<i64>::zeros(n);

    for (i, (img, label)) in images.iter().zip(labels.iter()).enumerate() {
        image_array.slice_mut(s![i, .., .., ..]).assign(img);
        label_array[i] = *label;
    }

    let images_path = out.join(format!("{}_images.npy", split_name));
    let labels_path = out.join(format!("{}_labels.npy", split_name));

    write_npy(&images_path, &image_array)?;
    write_npy(&labels_path, &label_array)?;

    println!("已保存: {:?} 和 {:?}", images_path, labels_path);

    Ok(())
}
