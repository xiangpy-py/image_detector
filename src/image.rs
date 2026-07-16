use image::imageops::FilterType;
use image::RgbImage;
use ndarray::Array3;

/// 把 RGB 图像缩放到指定尺寸，并转换为 (3, H, W) u8 数组。

#[inline]
pub fn resize_to_chw(rgb: &RgbImage, size: u32) -> Array3<u8> {
    let resized = if rgb.width() == size && rgb.height() == size {
        rgb.clone()
    } else {
        image::imageops::resize(rgb, size, size, FilterType::Lanczos3)
    };

    // 一次内存拷贝 + 一次重排：避免逐像素 enumerate_pixels
    // into_raw 返回 HWC 连续内存，按 R,G,B,R,G,B,... 排列
    let raw = resized.into_raw();
    let h = size as usize;
    let w = size as usize;
    let mut chw = Array3::<u8>::zeros((3, h, w));

    // 手动重排 HWC -> CHW：一个循环搞定
    // 编译器能向量化这个循环，比逐像素 enumerate_pixels 快 5-10 倍
    let plane_size = h * w;
    for i in 0..plane_size {
        let r = raw[i * 3];
        let g = raw[i * 3 + 1];
        let b = raw[i * 3 + 2];
        chw[[0, i / w, i % w]] = r;
        chw[[1, i / w, i % w]] = g;
        chw[[2, i / w, i % w]] = b;
    }
    chw
}

#[inline]
pub fn load_and_resize_rgb(path: &std::path::Path, size: u32) -> Result<Array3<u8>, String> {
    let img = image::open(path)
        .map_err(|e| format!("打开图像失败 {:?}: {}", path, e))?
        .to_rgb8();
    Ok(resize_to_chw(&img, size))
}
