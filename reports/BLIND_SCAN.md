# Quét độc lập trước khi xem pre-label

Frame: frame_0312.jpg tên một ảnh trong `to_label/round1/images/train/`

Số xe nhìn thấy bằng mắt: 23

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe: 
 - Có 2 xe con đi gần nhau và boundingbox khá trùng nhau ở gần chính giữa khung hình, đi trước xe tải và một xe con đen, mặc dù 2 xe đi không thật sự gần nhau, nhưng với góc nhìn này 2 xe dính vào nhau và dễ bị gán thành 1 xe
 - Xe ở góc dưới đường bên trái đã 1 nửa ra khỏi màn hình nhưng chưa khuất hẳn, mô hình dễ bỏ sót


Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
