# Lab Tracking — 2 giờ

Nhóm 2 người một máy. Detector đã khóa. Bạn chọn tracker và ngưỡng cho năm video khác cảnh.

## Việc cần làm

1. Tạo môi trường venv một lần tại thư mục gốc project. Cần Python 3.10:

~~~powershell
py -m pip install --user uv
py -m uv python install 3.10
py -m uv venv --seed --python 3.10 .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
git clone https://github.com/JonathonLuiten/TrackEval.git TrackEval
python -m pip install -e .\TrackEval\
python -m ipykernel install --user --name cv_robotics_lab21 --display-name cv_robotics_lab21
~~~

Những buổi chạy sau, mở terminal tại thư mục gốc project rồi kích hoạt môi trường:

~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

2. Tải ảnh năm video: [data_lab21.zip](https://drive.google.com/file/d/1UeVPQd6j5pSzxoJDcKJrerT9SL3vJLDt/view?usp=sharing). Giải nén, rồi đặt LAB_DATA thành thư mục chứa trực tiếp video_1 … video_5. Ví dụ nếu giải nén vào data_lab21\data_lab21:

~~~powershell
$env:LAB_DATA = (Resolve-Path .\data_lab21\data_lab21).Path
python scripts\check_data.py --lab-data-root $env:LAB_DATA
~~~

Cả năm video phải có ảnh. Chỉ video_1 có nhãn.

3. Mở on_tap_metrics.ipynb bằng kernel cv_robotics_lab21 từ terminal đã đặt LAB_DATA. Đọc bảng MOTA / IDF1 / HOTA, rồi chạy YOLO trên một ảnh video_1.

4. Chạy tracker. Bản thử có thể giới hạn frame. Bản nộp thì không.

~~~powershell
python scripts\run_tracking.py --source "$env:LAB_DATA\video_1\img1" --seq-name video_1 --tracker bytetrack --conf 0.3 --iou 0.5 --out runs\nop_bai --save-video
~~~

Đổi tên video và thư mục img1 cho video_2 … video_5. Tracker được chọn: bytetrack, ocsort, botsort, strongsort, deepocsort.

5. Chấm số chỉ video_1:

~~~powershell
python scripts\evaluate_practice.py --trackeval-root .\TrackEval --lab-data-root "$env:LAB_DATA" --submission runs\nop_bai\video_1.txt --run-name nhom01_video1
~~~

video_2 đến video_5 không có nhãn. Xem preview/video_N.mp4 trong thư mục dữ liệu và video có vẽ ID, rồi ghi điều bạn thấy.

## Luật chơi

| Khóa | Bạn chọn |
|---|---|
| Detector yolo26n.pt, ảnh 640 px, lớp người, Re-ID osnet_x0_25_msmt17.pt | Tracker, conf, iou của detector |

## Nộp

- video_1.txt … video_5.txt trong runs/nop_bai/ (đủ frame, đúng tên).
- submission_template/BAO_CAO_mau.md đã điền. Số HOTA / MOTA / IDF1 chỉ bắt buộc cho video_1.

Chi tiết từng bước, sự cố, và lịch 2 giờ: [HUONG_DAN.md](HUONG_DAN.md).
