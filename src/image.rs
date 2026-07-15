use std::path::Path;

use image::{imageops::FilterType, ImageError, ImageReader};
use ndarray::Array3;

/// 加载单张图像并缩放到指定尺寸。
///
/// 返回的数组形状为 (3, size, size)，按 RGB 通道优先存储。
pub fn load_and_resize(path: &Path, size: u32) -> Result<Array3<u8>, ImageError> {
    let img = ImageReader::open(path)?.decode()?;
    let img = img.resize_exact(size, size, FilterType::Lanczos3).to_rgb8();

    let mut tensor = Array3::<u8>::zeros((3, size as usize, size as usize));
    for (x, y, pixel) in img.enumerate_pixels() {
        for c in 0..3 {
            tensor[[c, y as usize, x as usize]] = pixel[c];
        }
    }

    Ok(tensor)
}
