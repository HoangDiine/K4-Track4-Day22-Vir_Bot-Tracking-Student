# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** ViR_Bot · **Thành viên:** Nguyễn Hoàng Duy - 2A202602751, Bui Quang Vinh - 2A20263012

Detector cố định: `yolo26n.pt`, ảnh 640 px, Re-ID `osnet_x0_25_msmt17.pt`. Không đổi các mục này trong bài nộp chính.

## CP1. Quan sát preview và giả thuyết

Đã kiểm tra năm clip gốc trong `data_lab21/data_lab21/preview/` qua các frame mẫu rải đều trong clip. Các dự đoán sau dùng để đối chiếu với video kết quả; đây là giả thuyết, chưa phải kết luận.

| Cảnh quan sát được | Giả thuyết cần kiểm tra |
|---|---|
| `video_1`: phố đi bộ ngoài trời, camera tĩnh, người đi thành nhóm và đôi lúc cắt ngang nhau. | ByteTrack có thể giữ ID ổn ở đoạn thoáng; lúc người đi sát hoặc che nhau có thể đổi ID. So sánh với StrongSORT để xem Re-ID có giúp không. |
| `video_2`: phố ban đêm, đông người, nhiều vùng sáng tối và người đi gần nhau. | Tracker thiên về chuyển động có thể lẫn người khi che khuất; Re-ID có thể giúp nếu còn nhìn rõ đặc điểm quần áo, nhưng ánh sáng yếu cũng có thể làm đặc trưng kém tin cậy. |
| `video_3`: camera di chuyển theo người đi bộ, ảnh ít chi tiết. | Chuyển động camera có thể làm ByteTrack mất liên kết giữa frame; Re-ID có thể giúp với người dễ phân biệt nhưng độ phân giải thấp có thể hạn chế tác dụng. |
| `video_4`: camera tiến trong trung tâm thương mại, sàn và kính phản chiếu, người đi cắt qua nhau. | Phản chiếu và che khuất có thể tạo hộp nhiễu hoặc đổi ID; cần xem liệu StrongSORT có giữ được người qua đoạn giao cắt hay không. |
| `video_5`: góc nhìn từ phương tiện ở phố/giao lộ, người đi bộ ở xa và có xe qua lại. | `conf` thấp có thể bắt thêm người nhỏ nhưng cũng tăng hộp nhiễu; `conf` cao có thể bỏ sót người ở xa. |

Notebook `on_tap_metrics.ipynb` chạy đủ bảng metric và câu True/False; YOLO phát hiện người trên một ảnh `video_1` nhưng chưa gán ID.

## CP2. Baseline ByteTrack trên video_1

Đã chạy ByteTrack với `conf=0.3`, `iou=0.5` trên 150 frame đầu. `runs/thu_nhanh/video_1_preview.mp4` có đúng 150 frame; `runs/thu_nhanh/video_1.txt` là kết quả thử nhanh, không dùng làm file nộp. Khi xem output ở frame 50, 100 và 145, ID 2, 3 và 5 vẫn xuất hiện trên cùng các track người; đây là đoạn giữ ID quan sát được trong bản baseline.

## 1. Cấu hình đã chọn

Mỗi video đã thử ByteTrack và StrongSORT với `conf=0.3`, `iou=0.5`; sau đó quét `conf=0.15/0.5` và `iou=0.4/0.7`, mỗi lượt chỉ đổi một tham số. Các lượt thử đều dùng 150 frame đầu. Bảng dưới ghi cấu hình chạy đủ frame và một cấu hình đã thử rồi loại.

Các giả thuyết CP1 dựa trên năm preview gốc. Nhận xét giữ ID trong bảng dưới đây dựa trên frame mẫu 50, 100 và 145 của preview kết quả tracking do script tạo.

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại |
|---|---|---|---|---|---|
| video_1 (phố đi bộ ngoài trời, tĩnh, ban ngày) | ByteTrack | 0.3 | 0.5 | Ở frame 50, 100, 145, ID 2, 3 và 5 còn gắn với cùng người. | `conf=0.15`: 674 dòng/9 ID; `conf=0.5`: 561/10; không cho thấy lợi ích rõ so với 0.3. |
| video_2 (phố đêm, tĩnh, rất đông) | StrongSORT | 0.3 | 0.5 | Cảnh tối, đông và nhiều người đi sát nhau; ID 1 vẫn theo người áo tối giữa hai frame mẫu. | ByteTrack 0.3/0.5: 1.313 dòng/16 ID; StrongSORT `conf=0.15`: 2.137/30, `conf=0.5`: 1.111/14. Chọn mức giữa để tránh tăng mạnh hoặc giảm mạnh số track. |
| video_3 (camera di động, ảnh nhỏ) | ByteTrack | 0.3 | 0.5 | Độ phân giải thấp, camera di chuyển; người áo sọc vẫn mang ID 1 ở hai frame mẫu. | StrongSORT 0.3/0.5 có 18 ID như ByteTrack nhưng chậm hơn; `conf=0.15` tăng lên 19 ID, `conf=0.5` còn 17. |
| video_4 (trong nhà, camera di chuyển) | StrongSORT | 0.3 | 0.5 | Camera tiến qua hành lang có kính; ID 2 (áo đỏ) và ID 6 (áo trắng) còn theo cùng người ở hai frame mẫu. | ByteTrack 0.3/0.5: 809 dòng/15 ID; StrongSORT `conf=0.15`: 1.147/31, `conf=0.5`: 778/12. Chọn mức giữa. |
| video_5 (trên xe bus, giao lộ đông) | ByteTrack | 0.3 | 0.5 | Góc nhìn rung và đổi theo xe; người ở xa nhỏ, các nhóm người thay đổi theo khung cảnh. | StrongSORT 0.3/0.5 tạo 950 dòng/30 ID so với ByteTrack 673/20; `conf=0.5` giảm còn 518 dòng. Số dòng/ID không chứng minh độ chính xác; giữ ByteTrack làm cấu hình thử nộp. |

## 2. Số liệu video_1

Dữ liệu không kèm `video_1/eval_config.json`; `scripts/evaluate_practice.py` đã được cập nhật để suy ra benchmark từ tên chuỗi trong `seqinfo.ini` và mặc định split là `train`. TrackEval chỉ chấm `video_1.txt`.

| HOTA | MOTA | IDF1 |
|---:|---:|---:|
| 26.912 | 17.292 | 25.713 |

```
TrackEval — video_1, cấu hình ByteTrack `conf=0.3`, `iou=0.5`, đủ 600 frame.
```

`video_2` đến `video_5` không có nhãn trong gói lab. Không điền số cho các video đó.

## 3. Phân tích

Quan sát video_1 có nhãn và các video còn lại bằng mắt; video_2 đến video_5 không có ground truth.

Ở video_1, ByteTrack giữ ID 2, 3 và 5 trên cùng người qua các frame mẫu 50, 100 và 145; kết quả này chỉ xác nhận đoạn đã xem, không khẳng định cả video 600 frame. TrackEval trên đủ video_1 cho HOTA 26.912, MOTA 17.292 và IDF1 25.713. Ở video_2, cảnh đêm đông làm người đi sát và che khuất nhau; StrongSORT ID 1 vẫn theo người áo tối giữa hai frame mẫu, nên chọn Re-ID để thử tiếp, nhưng không có nhãn để kết luận tracker tốt hơn ByteTrack. Ở video_4, ID 2 trên áo đỏ và ID 6 trên áo trắng còn theo cùng người giữa hai frame mẫu trong hành lang có phản chiếu; đây là dấu hiệu quan sát được, chưa phải kết quả định lượng. Ở video_5, góc quay từ xe bus rung và đổi liên tục, người ở xa nhỏ; ByteTrack được giữ làm lựa chọn thử, còn độ ổn định ID cần nhóm tiếp tục xem trong video đầy đủ.

## 4. Nếu có thêm thời gian

`conf=0.15` thường làm số dòng/ID tăng, rõ nhất ở video_2 và video_4; `conf=0.5` làm số dòng giảm mạnh, nên hai mức này bị loại để giữ `conf=0.3`. Thử `iou=0.4` và `0.7` gần như không đổi số dòng/ID so với `0.5`; nếu có thêm thời gian, nhóm sẽ xem các đoạn che khuất dài ở video_2/video_4 và thử mức `conf` quanh 0.3. Số dòng/ID chỉ là tín hiệu để chọn đoạn cần xem, không phải số đo độ chính xác cho video không nhãn.
