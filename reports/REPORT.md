# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Tạ Quang Lộc

Công cụ gán nhãn đã dùng: CVAT + sửa trực tiếp file nhãn và kiểm tra bằng review log (không dùng AnyLabeling/SAM)

## 1. Dữ liệu và cách chia tập

Tập pool và tập test được chia theo trục thời gian có vùng đệm ở giữa vì video là chuỗi liên tục; các khung hình gần nhau có cùng góc máy, cùng đường đi, cùng điều kiện ánh sáng và cùng loại phương tiện. Nếu random split, các frame kề nhau và gần như trùng cảnh sẽ rải vào cả train và test, khiến việc đánh giá AP50 bị quá lạc quan. Theo cách này, mô hình sẽ “nhìn thấy” các cảnh tương tự trong test và học thuộc kiểu ảnh hơn là tổng quát hóa; do đó, AP50 đo được sẽ cao hơn thực tế và không phản ánh khả năng vận hành trên dữ liệu mới. Nói ngắn gọn, chia theo thời gian giữ được tính chất “thực tế” của sự thay đổi cảnh và tránh rò rỉ gần trùng giữa train/test.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 trong `rounds_table.md` là:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Mô hình khởi đầu lạnh không khớp nhãn tham chiếu chủ yếu ở các xe nhỏ, xe ở góc khung hình và các xe bị che hoặc trùng gần nhau; đây là kiểu đối tượng mà pretrained COCO không có đủ dữ liệu để phát hiện ổn định trong cảnh giao thông. Recall theo kích thước cho thấy rõ: xe nhỏ chỉ 0.182, xe vừa 0.547, xe lớn 0.561. Điều này nói lên rằng khả năng phát hiện của model kém nhất trên xe nhỏ, còn xe lớn/xe vừa cũng chưa tốt. Một trường hợp cần rà lại nhãn tham chiếu trước khi kết luận model sai là khi xe nằm ở góc hoặc chỉ hiển thị một nửa thân xe, như trong BLIND_SCAN.md có cảnh “xe ở góc dưới đường bên trái đã 1 nửa ra khỏi màn hình nhưng chưa khuất hẳn”; ở dạng này, nhãn tham chiếu có thể bị mơ hồ và nên được kiểm lại theo guideline trước khi quy kết model phát hiện sai.

## 3. Chiến lược chọn mẫu

Chiến lược chọn mẫu theo `tools/al_select.py` là: `score = W_U·U + W_A·A + W_D·D` với `W_U=0.5`, `W_A=0.3`, `W_D=0.2`. Ở đây, `U` đo mức bất định của các box khó nhất trong frame, `A` đo tỉ lệ box mơ hồ so với max của pool, và `D` là độ đa dạng theo thời gian, nhằm tránh chọn các frame quá gần nhau về thời điểm. `MIN_GAP_S` là khoảng cách tối thiểu giữa hai frame được chọn, ở đây là 2 giây, để loại bỏ các frame gần như cùng cảnh vì cả hai gây chi phí rà nhãn tương đương nhưng mang ít giá trị học tập. Trong `SELECTION.md`, các ví dụ mạnh là `frame_0182.jpg` (rank 1, score 0.9591, 72.8s), `frame_0369.jpg` (rank 2, score 0.9324, 147.6s) và `frame_0326.jpg` (rank 4, score 0.9155, 130.4s): ba frame này đều có điểm cao, không quá gần trùng về thời gian, và nằm trong lô 12 ảnh được model chọn. Một frame điểm cao nhưng không chọn là `frame_0372.jpg` (rank 6, score 0.9101), vì nó nằm ngay sát các frame cuối chuỗi `0369`/`0380` nên gần trùng, và chi phí rà nhãn lớn hơn giá trị học tập. Còn `frame_0187.jpg` (rank 10, score 0.8995) dù không nằm top 5, vẫn đáng xem nếu ngân sách còn dư bởi vì nó có thời điểm khác và có thể bổ sung độ biến thiên cảnh. Điểm bất định cho thấy frame đó có nhiều box mơ hồ và khả năng sẽ cải thiện mô hình, nhưng chỉ khi frame đó không quá gần với các frame đã chọn; nếu gần trùng thì model học thêm rất ít dù có nhãn tốt. Nói cách khác, không chọn theo score đơn thuần, mà chọn theo “điểm cao + đa dạng thời gian + giảm trùng/chi phí”.

## 4. Các vòng học chủ động (active learning)

Bảng từ `rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 346 | 0.498 | -0.274 | 1.000 | 0.067 | 0.126 | 0.000 | 0.057 | 0.244 |

Sau khi sửa nhãn gợi ý trong vòng 1, `round1_diff.md` ghi nhận: model đề xuất 169 box, sau khi sửa còn 346 box; số box giữ nguyên 138, chỉnh sửa 19, xoá 12 (FP của model), thêm mới 189 (FN của model), accept rate 82%. Điều này chứng tỏ rằng dữ liệu chọn được không hề “dễ” và pre-label tốt; nó còn thiếu nhiều xe, đặc biệt xe gần nhau và xe ở góc. So với cold start, AP50 giảm từ 0.771 xuống 0.498, tương ứng -0.274, nghĩa là hiệu năng trên test giảm mạnh. Precision tăng từ 0.925 lên 1.000 nhưng recall sụt từ 0.489 xuống 0.067, F1 giảm từ 0.640 xuống 0.126. Cụ thể, recall small từ 0.182 xuống 0.000, medium 0.547 xuống 0.057, large 0.561 xuống 0.244; nhóm xe nào xấu đi rõ nhất là xe nhỏ và xe vừa. Không có nhóm nào tốt lên; cả ba nhóm đều tệ hơn sau fine-tune. Đây là một tín hiệu quan trọng: dữ liệu thêm vào bị thiếu nhãn và nghiêng về cảnh khó, nên fine-tune không bền vững.

Một ca thay đổi rõ sau fine-tune là `frame_0312.jpg`. Trong `BLIND_SCAN.md`, quan sát độc lập cho thấy “Số xe nhìn thấy bằng mắt: 23”, với 2 xe dính nhau ở giữa khung hình và 1 xe ở góc dưới trái bị cắt nửa khung. Trong `round1_diff.md`, model đề xuất 13 box nhưng sau sửa cần 22 box, với 8 được giữ nguyên, 3 chỉnh sửa, 2 xoá và 11 thêm mới. Đây là ví dụ điển hình cho việc mô hình sau train vẫn bỏ sót nhiều xe khó, đặc biệt xe dính nhau và xe ở mép ảnh. `REVIEW_LOG.csv` cũng cho thấy các sửa sai cụ thể: thêm xe ở góc dưới/trung tâm, chỉnh khung ôm sát thân xe gần góc; đây là lỗi pre-label đã sửa, không phải nhãn test “thực sự”. Tức là quan sát độc lập và review log là căn cứ để xác định ai đúng, còn output mô hình sau train chỉ là một gợi ý cần rà lại, không phải “ground truth”. Một ca khó theo guideline là xe ở góc dưới trái chưa khuất hẳn, xe dính nhau phía giữa và light/background tương đồng, vì đây là nơi khó phân tách từng đối tượng theo khung hình và dễ bị model bỏ sót hoặc gộp thành 1 box.

## 5. Kết luận và giới hạn

Kết quả vòng này tệ hơn cold start: AP50 từ 0.771 xuống 0.498, recall từ 0.489 xuống 0.067, F1 từ 0.640 xuống 0.126. Vì lý do này, tôi sẽ dừng hoặc không train thêm ngay mà kiểm tra lại chất lượng nhãn trước khi thực hiện vòng tiếp. Data thêm vào từ 12 frame có 346 box là quá ít và mang nhiều trường hợp khó, khiến mô hình sau fine-tune chủ yếu học “bộ phận dễ” và bỏ qua xe nhỏ, xe dính nhau và xe mép ảnh. Hai ca còn yếu hoặc bất định cho vòng sau là `frame_0331.jpg` (n_boxes=47, n_ambiguous=18, score 0.9154) và `frame_0372.jpg` (score 0.9101 nhưng `selected=False` vì gần trùng với `0369`/`0380`, nguy cơ không tăng thêm giá trị học tập). Cả hai đều có chi phí rà nhãn cao: 0331 là cảnh dày đặc, cần rà nhiều nhãn; 0372 là cảnh “điểm cao nhưng gần trùng”, cần cân nhắc nếu muốn tăng độ đa dạng hay chỉ xem như cảnh có thông tin thừa. Nếu AP50 giảm, trước khi train thêm tôi sẽ kiểm tra: (i) nhãn pre-label và xem có thiếu/không đúng guideline không; (ii) liệu có frame gần trùng hoặc duplicate quá nhiều trong batch không; (iii) có phải các frame chọn là hard-case mà model đã bị “lệch” do nhãn sai không; (iv) kiểm compare_round1.jpg và diff để xác định xem lỗi tập trung ở xe nhỏ, xe góc hay xe dính nhau. Tập test chỉ có 20 ảnh, thêm vào đó có luật bỏ qua xe dưới 16 px và nhãn test do mô hình tạo chưa được rà thủ công, nên kết luận về chất lượng mô hình cần giữ ở mức “có bằng chứng nhưng chưa đủ chặt chẽ”. Điều này không cho phép kết luận rằng model thực sự xấu hoặc tốt trên toàn bộ dữ liệu, mà chỉ cho thấy ở vòng này, dữ liệu chọn và quy trình rà nhãn cần được sửa trước khi tiếp tục.
