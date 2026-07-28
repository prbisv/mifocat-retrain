# MIFOCAT Code Breakdown

Ringkasan terstruktur tiap skrip utama di repo (preprocessing, prediksi, evaluasi). Fokus pada tujuan, alur inti, dan catatan penting.

## 1. load_dataset_v1.py
- **Peran:** Preprocessing sederhana MRI 3D ACDC (ED/ES) ke bentuk tetap 128³ + one-hot mask, simpan .npy.
- **Inti alur:**
  - Import: numpy, nibabel, glob, to_categorical, matplotlib, zoom, MinMaxScaler.
  - `resize_volume(data, new_shape=(128,128,128))` memakai `zoom(..., mode='nearest')` (aman untuk mask).
  - Contoh pasien: load ED/ES, normalisasi MinMax per-volume (reshape ke 2D untuk scaler), cast mask ke uint8, cek `np.unique`.
  - Visualisasi slice sebelum/ sesudah resize untuk sanity check.
  - Resize → expand channel → (128,128,128,1); mask one-hot 5 kelas.
  - Loop batch ED testing: load, normalisasi, resize, filter mask kosong (<1% non-bg), one-hot, simpan .npy (image dan mask); blok ES dikomentari.
- **Catatan:** Fixed shape memudahkan CNN 3D; filter mencegah sampel kosong.

## 2. load_dataset.py
- **Peran:** Preprocessing 3D berbasis spacing (SimpleITK) dengan multiprocessing; simpan satu file pat_XXX.npy per pasien (ed, ed_seg, es, es_seg).
- **Inti alur:**
  - Import: SimpleITK, multiprocessing.pool, pickle5, numpy, skimage.resize.
  - `resize_image(image, old_spacing, new_spacing, order)`: hitung `new_shape` dari rasio spacing (fisik konsisten).
  - Label mapping patologi: string ↔ indeks.
  - `convert_to_one_hot(seg)` untuk amankan resize mask.
  - `preprocess_image(itk_image, is_seg=False, spacing_target=(1,0.5,0.5), keep_z_spacing=False)`: ambil spacing (dibalik ke z,y,x), opsi tahan spacing z, citra distandardisasi, mask di-one-hot lalu resize order=1 → argmax.
  - `process_patient`: baca frame ED/ES, preprocess dengan target spacing (10,1.25,1.25), susun ke array 4 entri, simpan.
  - `generate_patient_info`: baca Info.cfg (frame ED/ES, height, weight, pathology).
  - `run_preprocessing`: pool(8) untuk ids range(101), simpan patient_info.pkl.
- **Catatan:** Bug potensial: memakai cPickle tapi tidak diimport; fungsi didefinisikan ulang jika dipanggil berulang.
- **Perbedaan dengan v1:** spacing-based vs shape-based; standardisasi vs MinMax; output per-pasien vs per-fase; paralel; tanpa filter mask kosong.

## 3. cek kelas groundtruth.py
- **Peran:** Memeriksa nilai unik mask PNG dan memisahkan 3 kelas ke mask biner terpisah.
- **Alur:** baca mask grayscale → `np.unique` → buat kelas1/2/3 via `np.where` (asumsi unique_values[0] background) → simpan gt_rv/gt_myo/gt_lv → plot 4 panel.
- **Catatan:** Mapping kelas bergantung urutan unique_values (rawan tertukar jika label berubah); aman jika label eksplisit ditetapkan.

## 4. cek npy.py
- **Peran:** Sanity check data .npy (train images/masks) dengan visualisasi slice acak.
- **Alur:** list dir images/masks → pilih indeks acak → load image (128³×1) dan mask one-hot → argmax mask → pilih slice z acak → plot image vs mask.
- **Catatan:** Banyak import tidak terpakai; label subplot “Image flair” hanya teks; pastikan urutan file sinkron (sebaiknya sort).

## 5. combined_images.py
- **Peran:** Overlay prediksi per kelas ke citra 2D dan simpan per pasien.
- **Alur:**
  - `create_folder` buat `saved_directory/focalctc/Pasien <id>`.
  - Natural sort untuk urutan slice.
  - `combined_images(...)`: validasi patient id via regex di nama file; baca image + mask grayscale; konversi ke BGR; buat boolean mask; warnai (saat ini hanya MYO aktif); simpan overlay per-slice.
  - `main`: loop pasien 101–150, set direktori input (images, prediksi myo/rv/lv), cek eksistensi, sort, panggil overlay.
- **Catatan:** RV/LV overlay dikomentari; pairing slice hanya via zip (akan diam pada list terpendek); warna dalam BGR.

## 6. custom_datagen.py
- **Peran:** Loader .npy sederhana dan generator batch untuk Keras.
- **Alur:**
  - `load_img`: iterasi nama file `.npy`, np.load, kumpulkan jadi array.
  - `imageLoader`: loop tak hingga; slicing list per batch_start/batch_end, load X/Y, yield (X,Y).
- **Catatan:** Tidak ada shuffle; os diimport tapi tak dipakai; concat path pakai `+` (lebih aman os.path.join); batch terakhir bisa lebih kecil.

## 7. hitung_dice_2d.py
- **Peran:** Dice coefficient untuk mask biner 2D (bisa 3D asal biner).
- **Alur:** intersection = sum(gt*pred); handle sum 0 → return 0; dice = 2*intersection/(sum_gt+sum_pred); guard NaN.
- **Catatan:** Untuk multi-class, hitung per kelas (biner) lalu rata-rata; keputusan kosong-kosong → 0 (bukan 1).

## 8. hitung_evaluasi_metrik.py
- **Peran:** Evaluasi slice-wise pasien tunggal (i=104, fase ED) untuk MYO/RV/LV terhadap GT PNG; metrik IoU, Dice, Hausdorff, mean surface distance; simpan CSV + boxplot.
- **Alur:**
  - Path prediksi/GT per kelas; output CSV di folder pasien.
  - Tiap fungsi kelas: validasi jumlah file, load PNG, biner 255→1, hitung metrik, tulis CSV (desimal koma), simpan list untuk rata-rata.
  - `plot_visualize`: boxplot IoU/Dice/Hausdorff/Surface (MYO vs RV vs LV), simpan PNG.
  - `main`: list & sort file per kelas, panggil evaluasi tiap kelas, lalu plotting.
- **Catatan:** Pairing hanya berdasarkan urutan list (lebih aman pakai nama slice); impor load_model/LabelEncoder/tf utils tidak dipakai; kode duplikat antar kelas.

## 9. hitung_evaluasi_metrik_acdc2017.py
- **Peran:** Versi evaluasi massal (pasien 101–150) untuk prediksi TransUNet; struktur mirip file #8 tetapi di-loop per pasien.
- **Alur:** Dalam loop pasien: definisikan path prediksi/GT, fungsi evaluasi per kelas (sama pola: biner 255→1, IoU/Dice/Hausdorff/Surface, CSV, mean), fungsi plot boxplot, `main()` dipanggil per pasien.
- **Perbedaan utama vs #8:** banyak pasien, path model TransUNet, output penamaan transunet; fungsi didefinisikan berulang (kurang efisien tapi jalan).

## 10. hitung_f1score_2d.py
- **Peran:** F1-score biner 2D dengan perhitungan TP/FP/FN pixel-wise.
- **Alur:** (Opsional) konversi 255→1 dikomentari; loop dua tingkat hitung TP/FP/FN; F1 = tp / (tp + 0.5*(fp+fn)); jika tidak ada kasus → 0.
- **Catatan:** Lambat untuk banyak slice; input harus 0/1; print shape menambah noise; kosong-kosong → 0.

## 11. hitung_iou_2d.py
- **Peran:** IoU biner 2D, implementasi manual TP/FP/FN (loop pixel).
- **Alur:** (Opsional) konversi 255→1 dikomentari; hitung TP/FP/FN; IoU = tp/(tp+fp+fn) else 0; debug print shape.
- **Catatan:** Lebih cepat pakai versi vektorisasi (sudah ada di komentar); pastikan input biner dan shape sama; kosong-kosong → 0.

## 12. hitung_mcc_2d.py
- **Peran:** Matthews Correlation Coefficient untuk mask biner 2D (vektorisasi numpy).
- **Alur:** hitung TP/TN/FP/FN via mask boolean; numerator = tp*tn - fp*fn; denominator = sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn)+1e-7); jika denom 0 → 0; else numerator/denominator.
- **Catatan:** Input harus 0/1 (konversi 255→1 jika perlu); metrik robust untuk dataset imbalanced.

## 13. hitung_surface_distance_2d.py
- **Peran:** Hitung symmetric surface distance antara dua mask biner.
- **Alur:** cast ke bool → definisikan struktur konektivitas → boundary = mask XOR erode(mask) → distance transform (~boundary) → kumpulkan jarak boundaryA→B dan B→A; jika kosong → 0; return array jarak.
- **Catatan:** Sampling dapat diisi spacing fisik; input multi-class akan digabung foreground; return dapat int 0 jika kosong.

## 14. pisah_gt_myo.py
- **Peran:** Ekstrak kelas MYO dari mask PNG multiclass menjadi mask biner (0/255).
- **Alur:** list file GT, sort by mtime; untuk tiap PNG: baca, `np.unique`; jika len==3 ambil unique[1], jika len==4 ambil unique[2], else kosong; simpan ke folder gt_myo (nama sama).
- **Catatan:** Mapping kelas berbasis urutan unique_values (tidak eksplisit label); tidak membuat folder output; teks log menyebut resize tapi hanya pemisahan.

## 15. predict_trans_unet.py
- **Peran:** Inferensi TransUNet 2D (ED) dengan banyak custom layer; simpan mask gabungan dan biner per kelas (RV/MYO/LV).
- **Alur:**
  - Definisikan custom layers (patch extract/embedding, patch merging/expanding, window attention, Swin blocks, drop_path, GELU, Snake, dll.).
  - `membuat_direktori_2d_ed_prediksi` struktur output per pasien (citra prediksi, prediksi RV/MYO/LV).
  - `prediksi_citra`: regex ambil id pasien; baca PNG grayscale; ulang channel jadi 3 (model dilatih 3 channel); model.predict; argmax label map; simpan gabungan (dinormalisasi 0–255 untuk visual) dan mask biner tiap kelas (jika kelas tidak muncul → kosong).
  - `main`: set custom_objects, load model (compile=False), ambil semua citra ED, jalankan prediksi.
- **Catatan:** Custom layer wajib didefinisikan saat load; normalisasi gabungan hanya untuk visual (bukan evaluasi); jaga div0 jika max=0.

## 16. predict_unet_2d.py
- **Peran:** Inferensi U-Net 2D (ED) grayscale 1-channel; simpan gabungan + mask biner per kelas.
- **Alur:**
  - Utility direktori output per pasien.
  - `prediksi_citra`: regex id pasien; baca grayscale; expand channel ->1; norm 0–1; model.predict; argmax; simpan gabungan (0–255) dan mask biner RV/MYO/LV (kosong jika kelas absen).
  - `main`: load model compile=False; glob semua PNG ED; jalankan prediksi.
- **Catatan:** Import tak terpakai (LabelEncoder, to_categorical); fungsi get_modification_time tidak dipakai; guard jika predicted_img.max()==0 sebaiknya ada.

## 17. slice3dto2d.py
- **Peran:** Pipeline konversi NIfTI 3D → PNG 2D (ED/ES), resize 128×128, dan split GT kelas (testing ED).
- **Alur:**
  - Buat struktur folder ED/ES (train/testing) termasuk GT kelas untuk testing.
  - `slice_ed`: load volume+mask; loop slice; normalisasi ke 0–255; simpan PNG (citra+GT) tanpa resize.
  - `slice_es`: serupa, citra/GT diproses terpisah.
  - `resize_img_ed` / `resize_img_es`: resize citra+GT ke 128×128 (GT INTER_NEAREST) untuk train.
  - `resize_img_ed_test`: resize testing ED; cek unique; pisah kelas berdasarkan nilai 85(RV)/170(MYO)/255(LV) ke mask biner 0/255; simpan per pasien (images + GT per kelas).
  - `main`: pilih tahapan (saat ini aktif resize_img_ed_test).
- **Catatan:** Fondasi data 2D; jika label encoding berubah, pemisahan kelas harus disesuaikan.

## 18. split_data.py
- **Peran:** Split folder dataset ke train/val memakai splitfolders (default 90/10).
- **Alur:** set input_folder/output_folder; panggil `splitfolders.ratio(..., seed=42, ratio=(.90,.10))` menjaga struktur kelas.
- **Catatan:** Untuk tambah test set, ubah ratio tuple; pastikan konsistensi images/masks jika segmentasi.

## 19. test_gambar.py
- **Peran:** Pipeline evaluasi U-Net 3D (ED): (prediksi 3D → slice 2D → metrik Myo Dice & Hausdorff → CSV). Prediksi/slicing dikomentari, evaluasi aktif.
- **Alur:**
  - Buat direktori GT dan prediksi (per pasien, dengan varian keseluruhan/RV/MYO/LV).
  - `slice_gt`: argmax GT 3D → simpan slice keseluruhan + Myo.
  - `prediksi_ed`: model.predict pada citra 3D .npy; argmax; simpan slice prediksi per kelas.
  - `hitung_evaluasi_metrik`: load prediksi Myo vs GT Myo PNG; biner 255→1; hitung Dice & Hausdorff per slice; tulis CSV (mean di akhir).
  - `main`: load model, list file, sort by mtime, panggil evaluasi (prediksi/slicing tetap dikomentari).
- **Catatan:** Evaluasi hanya Myo; pairing file via waktu modifikasi; delimiter CSV `;`.

## 20. hausdorff/distances.py
- **Peran:** Implementasi metrik jarak (manhattan, euclidean, chebyshev, cosine, haversine) dengan numba JIT untuk kecepatan; digunakan oleh metrik boundary (Hausdorff/surface).
- **Alur:** Set decorator `@numba.jit(nopython=True, fastmath=True)`; tiap fungsi terima dua vektor sama panjang dan hitung jarak sesuai metrik.
- **Catatan:** Input harus numeric dengan panjang sama; fastmath bisa sedikit kompromi presisi; cosine/haversine tidak dipakai untuk mask tetapi ada untuk general utility.

