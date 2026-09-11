# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy: 11/09/2026**

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):

"class_id": 468, "class_name": "cab", "rank": 1,"score": 0.510915, "taxonomy_name": "ImageNet-1K".

- Record này mô tả toàn ảnh như thế nào?

Mỗi record phân loại mô tả toàn bộ hình ảnh bằng cách gán một nhãn lớp như cab, minibus. Các nhãn được sắp xếp theo score của mô hình, cho biết mức độ tin cậy của model vào việc dự đoán đó cho toàn bộ hình ảnh.

- Ai định nghĩa class list mà checkpoint có thể dự đoán?

Class list được định nghĩa bởi tập dữ liệu ImageNet-1K mà mô hình được train, qua đó định nghĩa các lớp mà model có thể dự đoán.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?

class_id là số nguyên duy nhất đại diện cho mỗi lớp, giúp dễ dàng xử lý và lưu trữ hiệu quả. class_name là tên giúp người dễ đọc, phân tích và xử lý (cab, minibus), giúp người không phải tra cứu id mỗi lần. taxonomy_name chỉ ra nguồn gốc của lác lớp có cùng tên nhưng thuộc về các dataset/ngữ cảnh khác nhau, nó cung cấp ngữ cảnh cho class_id, class_name.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?

Với ảnh có nhiều chủ thể, guideline cần quy định rõ cách chọn một nhãn duy nhất cho toàn bộ ảnh, hoặc cách ưu tiên các vật thể khi gán nhãn.
Quy tắc ưu tiên: Chọn lớp của vật thể lớn nhất, nổi bật nhất, hoặc vật thể ở trung tâm. Quy tắc số lượng: Chọn lớp của loại vật thể có số lượng nhiều nhất. Quy tắc mục đích: Chọn lớp của vật thể phù hợp nhất với mục đích sử dụng của dữ liệu. Khuyến khích đa nhãn: Nếu yêu cầu là mô tả toàn diện, guideline có thể cho phép gán nhiều nhãn cho một ảnh, với hướng dẫn về cách sắp xếp thứ tự hoặc giới hạn số lượng nhãn.

- Vì sao model score không phải ground truth?

Model score là giá trị model đưa ra, thể hiện độ tự tin về một dự đoán cụ thể, nó không phải ground truth vì : (1) Không phải dữ liệu thực tế, ground truth là sự thật khách quan, là nhãn chính xác được người xác định dựa trên định nghĩa/guideline. Model score chỉ là một ước tính của model. (2) model có thể có độ tin cậy cao vào 1 dự đoán sai (FP), hoặc độ tin cậy thấp vào 1 dự đoán đúng (FN). (3) Model score bị ảnh hưởng bởi kiến trúc model, data training và phương pháp traning. Nó có thể thay đổi giữa các phiên bản khác nhau, trong khi ground truth là cố định.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):

"class_name": "bus", "score": 0.912558,
"bbox_xyxy": [
93.17,
187.95,
223.01,
320.91
],  
"bbox_width": 129.84,
"bbox_height": 132.96

- Diễn giải vị trí box bằng lời:

Record cho biết model đã phát hiện một vật thể thuộc lớp bus với độ tin cậy 0.912558. Vật thể này nằm trong vùng hình chữ nhật, bắt đầu từ điểm (x_min, y_min) = (93.17, 187.95) và kết thúc tại điểm (x_max, y_max)=(223.01, 320.91). Hộp có chiều rộng là 129.84 pixel và chiều cao là 132.96 pixel.

- So sánh số prediction ở hai threshold:

Với threshold = 0.20: 17 vật thể được phát hiện.
Với threshold = 0.35: 11 vật thể được phát hiện.
Với threshold = 0.60: 6 vật thể được phát hiện.

Khi ngưỡng threshold giảm từ 0.35 xuống 0.20, số lượng vật thể được dự đoán tăng lên từ 11 lên 17. Ngược lại, khi ngưỡng tăng từ 0.35 lên 0.60, số lượng vật thể được dự đoán giảm đi từ 11 xuống 6. cho thấy rằng việc giảm ngưỡng cho phép model chấp nhận các dự đoán có độ tin cậy thấp hơn, nhiều phát hiện hơn, nhưng cũng có thể nhiều lỗi hơn (FP).

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?

Độ bao phủ: Khi ngưỡng thấp hơn, độ bao phủ của mô hình sẽ tăng lên. model có khả năng phát hiện được nhiều vật thể thực tế hơn trong ảnh (tăng recall), bao gồm cả những vật thể mà nó không quá tự tin. Tuy nhiên, điều này cũng có thể bao gồm nhiều dự đoán sai.

Khối lượng reviewer cần xem: Khi số lượng dự đoán tăng lên do ngưỡng thấp, khối lượng công việc cho reviewer (người gán nhãn/kiểm tra chất lượng) sẽ tăng đáng kể. Họ sẽ phải xem xét và xác minh nhiều hộp giới hạn hơn, bao gồm cả những hộp có độ tin cậy thấp có thể là false positives, để đảm bảo chất lượng ground truth.

- Đề xuất một quy tắc box chặt:

Hộp giới hạn phải ôm sát vật thể được gán nhãn, chỉ bao gồm các pixel thuộc vật thể đó và không bao gồm các khoảng trống đáng kể giữa vật thể và hộp. Không được để hộp giới hạn chồng chéo với các vật thể không liên quan hoặc để các phần rõ ràng của vật thể bị cắt bỏ bởi hộp giới hạn. Đối với các vật thể có hình dạng phức tạp, hãy cố gắng định vị hộp sao cho diện tích không phải vật thể bên trong hộp là tối thiểu.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?

Guideline về che khuất
Cần quy định một tỷ lệ phần trăm tối thiểu của vật thể phải hiển thị để được gán nhãn.
Làm rõ việc gán nhãn cho vật thể bị che khuất nhưng vẫn có thể dễ dàng nhận dạng bởi con người.
Hộp giới hạn nên vẽ toàn vật thể (tự đoán) hay chỉ phần hiển thị.

Guideline về cắt mép ảnh
Vật thể chỉ xuất hiện một phần ở rìa ảnh có nên được gán nhãn không.
Hộp giới hạn nên vẽ đến rìa hay đến ước tính toàn bộ kính thước vật thể.

Escalation
Các trường hợp phức tạp, mơ hồ hay vật bị che khuất khó xác định, cần trao đổi với leader để đưa ra quyết định cuối, đảm bảo nhất quán cho dataset.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):

  "instance_id": "traffic-001", "class_name": "bus", "score": 0.925745, "polygon_point_count": 120,
  "polygon_xy": [
  [
  148.0,
  189.0
  ],
  [
  147.0,
  190.0
  ],
  [
  145.0,
  190.0
  ],
  [
  143.0,
  192.0
  ],
  [
  142.0,
  192.0
  ],
  [
  141.0,
  193.0
  ],
  [
  139.0,
  193.0
  ],
  [
  137.0,
  195.0
  ],
  [
  136.0,
  195.0
  ],
  [
  135.0,
  196.0
  ],
  [
  134.0,
  196.0
  ],
  [
  133.0,
  197.0
  ],
  [
  132.0,
  197.0
  ],
  [
  131.0,
  198.0
  ],
  [
  130.0,
  198.0
  ],
  [
  128.0,
  200.0
  ],
  [
  127.0,
  200.0
  ],
  [
  126.0,
  201.0
  ],
  [
  125.0,
  201.0
  ],
  [
  123.0,
  203.0
  ],
  [
  122.0,
  203.0
  ],
  [
  121.0,
  204.0
  ],
  [
  120.0,
  204.0
  ],
  [
  119.0,
  205.0
  ],
  [
  117.0,
  205.0
  ],
  [
  116.0,
  206.0
  ],
  [
  115.0,
  206.0
  ],
  [
  114.0,
  207.0
  ],
  [
  113.0,
  207.0
  ],
  [
  111.0,
  209.0
  ],
  [
  110.0,
  209.0
  ],
  [
  109.0,
  210.0
  ],
  [
  108.0,
  210.0
  ],
  [
  105.0,
  213.0
  ],
  [
  104.0,
  213.0
  ],
  [
  98.0,
  219.0
  ],
  [
  98.0,
  220.0
  ],
  [
  96.0,
  222.0
  ],
  [
  96.0,
  312.0
  ],
  [
  97.0,
  313.0
  ],
  [
  97.0,
  314.0
  ],
  [
  98.0,
  315.0
  ],
  [
  101.0,
  315.0
  ],
  [
  102.0,
  316.0
  ],
  [
  104.0,
  316.0
  ],
  [
  105.0,
  317.0
  ],
  [
  106.0,
  317.0
  ],
  [
  107.0,
  318.0
  ],
  [
  108.0,
  318.0
  ],
  [
  109.0,
  319.0
  ],
  [
  113.0,
  319.0
  ],
  [
  114.0,
  318.0
  ],
  [
  116.0,
  318.0
  ],
  [
  117.0,
  317.0
  ],
  [
  122.0,
  317.0
  ],
  [
  123.0,
  316.0
  ],
  [
  131.0,
  316.0
  ],
  [
  132.0,
  315.0
  ],
  [
  163.0,
  315.0
  ],
  [
  164.0,
  314.0
  ],
  [
  170.0,
  314.0
  ],
  [
  171.0,
  315.0
  ],
  [
  176.0,
  315.0
  ],
  [
  177.0,
  316.0
  ],
  [
  183.0,
  316.0
  ],
  [
  184.0,
  315.0
  ],
  [
  185.0,
  315.0
  ],
  [
  188.0,
  312.0
  ],
  [
  188.0,
  311.0
  ],
  [
  190.0,
  309.0
  ],
  [
  190.0,
  308.0
  ],
  [
  197.0,
  301.0
  ],
  [
  198.0,
  301.0
  ],
  [
  199.0,
  300.0
  ],
  [
  201.0,
  300.0
  ],
  [
  202.0,
  299.0
  ],
  [
  203.0,
  299.0
  ],
  [
  204.0,
  298.0
  ],
  [
  205.0,
  298.0
  ],
  [
  206.0,
  297.0
  ],
  [
  209.0,
  297.0
  ],
  [
  210.0,
  296.0
  ],
  [
  211.0,
  296.0
  ],
  [
  213.0,
  294.0
  ],
  [
  213.0,
  293.0
  ],
  [
  214.0,
  292.0
  ],
  [
  214.0,
  291.0
  ],
  [
  215.0,
  290.0
  ],
  [
  215.0,
  289.0
  ],
  [
  217.0,
  287.0
  ],
  [
  217.0,
  286.0
  ],
  [
  219.0,
  284.0
  ],
  [
  219.0,
  283.0
  ],
  [
  220.0,
  282.0
  ],
  [
  220.0,
  281.0
  ],
  [
  221.0,
  280.0
  ],
  [
  221.0,
  279.0
  ],
  [
  222.0,
  278.0
  ],
  [
  222.0,
  276.0
  ],
  [
  223.0,
  275.0
  ],
  [
  223.0,
  234.0
  ],
  [
  222.0,
  233.0
  ],
  [
  222.0,
  226.0
  ],
  [
  221.0,
  225.0
  ],
  [
  221.0,
  220.0
  ],
  [
  220.0,
  219.0
  ],
  [
  220.0,
  208.0
  ],
  [
  219.0,
  207.0
  ],
  [
  219.0,
  202.0
  ],
  [
  218.0,
  201.0
  ],
  [
  218.0,
  198.0
  ],
  [
  217.0,
  197.0
  ],
  [
  217.0,
  196.0
  ],
  [
  213.0,
  192.0
  ],
  [
  211.0,
  192.0
  ],
  [
  210.0,
  191.0
  ],
  [
  206.0,
  191.0
  ],
  [
  205.0,
  190.0
  ],
  [
  199.0,
  190.0
  ],
  [
  198.0,
  189.0
  ]
  ]
  },

- Polygon bổ sung chi tiết gì so với box?

Polygon bổ sung cho box về hình dạng chính xác của vật thể. Nếu box là một hình chữ nhật bao quanh vật thể thì Polygon bám sát các đường cong, góc cạnh và các chi tiết phức tạp của vật thể.

- `instance_id` dùng để làm gì và không phải loại ID nào?

instance_id xác định từng vật thể duy nhất trong ảnh, cho phép theo dõi và phân tích từng vật thể riêng lẻ. instance_id không phải là class_id (định danh cho một loại vật thể), coco_image_id (id duy nhất cho toàn bộ ảnh).

- Đề xuất một quy tắc biên mask:

Mask phải theo sát đường viền nhìn thấy được của vật thể, kông để sót các pixel thuộc vật thể và không bao gồm pixel nền. Biên mask phải mịn, tránh các đường zic-zac không cần thiết. Đối với các vật thể có cấu trúc chi tiết nhỏ (tóc, lá cây), hãy vẽ theo đường bao chung, không cố gắng vẽ từng chi tiết cực nhỏ nếu chúng không rõ ràng. Biên mask phải được vẽ khép kín.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?

Yêu cầu guideline cần cụ thể và quy trình escalation:
Vùng mờ:
guideline cần quy định mức độ mờ cho phép để một vật thể vẫn được gán nhãn
Escalation: Trao đổi với leader trong trường hợp vật có đủ rõ để gán nhãn hay không.

Vùng tiếp xúc:
Guideline: Cần quy định cách vẽ đường biên mask khi 2 vật thể cùng loại/khác loại chạm vào nhau.
Escalation: Trao đổi với leader trong trường hợp đường biên giữa các vật thể tiếp xúc quá phức tạp hoặc mơ hồ.

Vùng che khuất:
Guideline: Cần quy định tỷ lệ % tối thiểu của vật thể phải hiển thị.
Escalation: Đưa ra phương pháp gán nhãn thống nhất trong trường hợp che khất phức tạp, hình dạng của vật thể bị che khuất khó ước tính.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ                | Đơn vị/định dạng ground truth                                                                           | Lỗi hoặc điểm mơ hồ quan sát được                                                                                                                                                 | Annotator làm gì?                                                                                                                                                                                                                                    | Reviewer xem gì?                                                                                                                                                                                                                                                         |
| --------------------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Phân loại ảnh         | Một nhãn duy nhất hoặc danh sách các nhãn cho toàn bộ ảnh cùng với ID, tên lớp và tên taxonomy.         | Lớp sai, ảnh có nhiều chủ thể nhưng chỉ gán một nhãn không đại diện đúng, độ tin cậy của mô hình không phải là ground truth                                                       | Đánh giá nội dung tổng thể của ảnh. Áp dụng các quy tắc guideline để chọn nhãn phù hợp nhất.                                                                                                                                                         | Đảm bảo nhãn được gán đúng với guideline (độ ưu tiên vật thể, tính đại diện). Kiểm tra tính nhất quán trong việc gán nhãn cho các ảnh tương tự. Xác minh class_id, class_name và taxonomy_name là chính xác theo danh sách lớp cho trước.                                |
| Phát hiện vật thể     | Một bounding box cùng với class_name và class_id cho mỗi vật thể trong ảnh                              | Bounding box bị lệch, quá rộng/hẹp, hoặc không bao phủ toàn bộ vật thể. Thiếu vật thể (model bỏ sót). Phát hiện sai (false positive). Vật thể bị che khuất/cắt mép khó xác định.  | Xác định vị trí và vẽ box giới hạn ôn sát mỗi vật thể nhìn thấy được trong ảnh. Tuân thủ các quy tắc về việc vẽ box chặt và xử lý vật thể bị che khuất/cắt mép.                                                                                      | Kiểm tra độ chính xác của từng bounding box ôm sát, không thừa/thiếu pixel. Đảm bảo mọi vật thể nhìn thấy được đều được gán nhãn. Phát hiện các false positive và false negative. Áp dụng guideline về vật bị che khuất/cắt mép                                          |
| Instance segmentation | Một polygon_xy (mask) chi tiết, class_name, class_id và instance_id cho mỗi phiên bản vật thể trong ảnh | Mask không theo sát đường biên vật thể, quá lớn/nhỏ, hoặc có lỗ hổng. Thiếu mask. Instance_id bị trùng lặp hoặc sai. Vùng mờ, vật thể tiếp xúc hoặc bị che khuất làm mask khó vẽ. | Xác định và vẽ mask theo sát đường viền chính xác của mỗi phiên bản vật thể trong ảnh. Gán class_id, class_name và instance_id duy nhất cho từng vật thể. Tuân thủ các quy tắc về biên mask và xử lý các trường hợp mơ hồ (che khuất, mờ, tiếp xúc). | Kiểm tra độ chính xác của từng mask: ôm sát, không thừa/thiếu pixel, mịn. Đảm bảo mọi phiên bản vật thể nhìn thấy được đều có mask. Xác minh instance_id là duy nhất cho từng vật thể. Đảm bảo tuân thủ guideline cho các trường hợp đặc biệt (mờ, che khuất, tiếp xúc). |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:

Nếu bạn thấy ảnh hoặc dữ liệu không đúng phạm vi (ảnh nhạy cảm, thông tin cá nhân không được phép, dữ liệu không liên quan đến dự án), bạn sẽ dừng công việc ngay lập tức và báo cáo cho leader.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  - Người quản lý dự án (PM): Đây là người chịu trách nhiệm chính về phạm vi và tính toàn vẹn của dữ liệu.
  - Trưởng nhóm gán nhãn (Team Lead): Họ có thể hướng dẫn về các bước tiếp theo và đảm bảo rằng không có dữ liệu không phù hợp nào được xử lý thêm.
  - Nhóm pháp lý hoặc bảo mật dữ liệu: Nếu dữ liệu vi phạm chính sách bảo mật hoặc có tính nhạy cảm cao, việc báo cáo cho nhóm này là cần thiết để đánh giá rủi ro và có biện pháp xử lý phù hợp.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
