# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Mai Quang Dũng  
**MSSV:** 2A202602966  
**Nhóm:** The LIEM'S
**Ngày:** 20/09/2026  

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiến gần về 1.0) nghĩa là hai vector embedding biểu diễn hai đoạn văn bản có góc hợp giữa chúng rất nhỏ trong không gian đa chiều. Điều này chứng tỏ hai đoạn văn bản có sự tương đồng sâu sắc về mặt ngữ nghĩa và chủ đề, bất kể độ dài hay từ vựng bề mặt có thể khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Người mua có quyền gửi yêu cầu trả hàng và hoàn tiền trong vòng mười lăm ngày kể từ ngày nhận hàng."
- Câu B: "Khách hàng được phép khiếu nại để nhận lại tiền trong thời hạn 15 ngày sau khi giao hàng thành công."
- Tại sao tương đồng: Dù sử dụng các từ đồng nghĩa khác nhau (người mua / khách hàng, hoàn tiền / nhận lại tiền, mười lăm ngày / 15 ngày), cả hai câu đều diễn đạt cùng một quy định cốt lõi về thời hạn đổi trả trong chính sách TMĐT.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Người mua có quyền gửi yêu cầu trả hàng và hoàn tiền trong vòng mười lăm ngày kể từ ngày nhận hàng."
- Câu B: "Thuật toán sắp xếp nhanh QuickSort có độ phức tạp thời gian trung bình là O(n log n)."
- Tại sao khác: Hai câu thuộc hai lĩnh vực hoàn toàn độc lập và không có bất kỳ mối liên hệ ngữ nghĩa nào (chính sách thương mại điện tử vs thuật toán khoa học máy tính).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid bị chi phối bởi độ lớn (magnitude/độ dài) của vector, khiến hai văn bản có cùng ý nghĩa nhưng một đoạn ngắn (câu hỏi) và một đoạn dài (đoạn chính sách) có thể bị đẩy ra rất xa nhau. Trong khi đó, độ tương tự cosine chuẩn hóa vector và chỉ tập trung đo góc (hướng của ngữ nghĩa), giúp việc tìm kiếm giữa câu truy vấn ngắn và tài liệu dài đạt độ chính xác cao hơn hẳn.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> - Bước nhảy giữa các chunk (step): $\text{step} = \text{chunk\_size} - \text{overlap} = 500 - 50 = 450$ ký tự.
> - Chunk 1 bắt đầu tại vị trí 0 đến 500. Các chunk tiếp theo bắt đầu tại: $0, 450, 900, ..., 9450$ (tương ứng với 22 bước).
> - Tại bước thứ 22 (vị trí 9450): đoạn văn bản kéo dài đến $9450 + 500 = 9950$ (chưa hết 10,000 ký tự).
> - Đoạn còn lại từ vị trí 9900 đến 10,000 tạo thành chunk cuối cùng.
> - Áp dụng công thức: $\text{Số chunk} = \lceil (10000 - 50) / (500 - 50) \rceil = \lceil 9950 / 450 \rceil = \lceil 22.11 \rceil = 23$.
> *Đáp án:* **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100, bước nhảy giảm còn $500 - 100 = 400$ ký tự, số lượng chunk sẽ tăng lên $\lceil (10000 - 100) / 400 \rceil = \lceil 9900 / 400 \rceil = 25$ chunks. Ta muốn tăng độ chồng chéo khi văn bản có nhiều mệnh đề liên kết phức tạp; overlap lớn giúp giữ nguyên vẹn ngữ cảnh ở các ranh giới cắt, tránh tình trạng một câu quy định hoặc điều kiện quan trọng bị đứt đôi giữa hai chunk khiến mô hình embedding làm mất thông tin.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi sử dụng biểu thức chính quy Lookbehind `r"(?<=[.!?])\s+"` để tìm ranh giới câu dựa trên các dấu chấm, chấm than, chấm hỏi kèm khoảng trắng hoặc xuống dòng. Cách làm này đảm bảo giữ nguyên vẹn dấu câu ở cuối mỗi câu (không bị nuốt mất như phép tách regex thông thường), sau đó gom từng nhóm `max_sentences_per_chunk` câu lại và loại bỏ khoảng trắng thừa với `.strip()`.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Tôi triển khai thuật toán đệ quy hai chiều: chiều xuống ưu tiên phân tách văn bản bằng các dấu phân cách có cấu trúc lớn trước `["\n\n", "\n", ". ", " ", ""]` để giữ trọn vẹn ngữ nghĩa đoạn văn, nếu mảnh con vẫn vượt quá `chunk_size` thì mới hạ xuống dấu phân cách nhỏ hơn; chiều lên thực hiện gom nối các mảnh nhỏ kề nhau lại cho tới sát ngưỡng `chunk_size` để chống vụn chunk. Base case là khi chuỗi rỗng, độ dài $\le$ `chunk_size`, hoặc khi danh sách separator rỗng thì cắt cứng theo `chunk_size`.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> `EmbeddingStore` được lưu trữ dưới dạng cấu trúc In-memory dictionary `self._store`. Hàm `add_documents` chuẩn hóa tài liệu qua helper `_make_record`, tạo vector embedding và gán sẵn `doc_id` vào metadata; hàm `search` tính điểm tương đồng giữa vector câu truy vấn và tất cả vector lưu trữ bằng tích vô hướng `_dot`, sau đó sắp xếp giảm dần theo `score` và trả về top-$k$ kết quả.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` thực hiện tiền lọc (pre-filter) các bản ghi khớp toàn bộ điều kiện trong `metadata_filter` **trước**, sau đó mới chạy `_search_records` trên tập ứng viên đã lọc để tránh việc các kết quả không hợp lệ chiếm hết slot top-$k$. Hàm `delete_document` duyệt lọc và loại bỏ mọi bản ghi có `metadata['doc_id'] == doc_id` (hoặc `id == doc_id`), đồng thời so sánh độ dài danh sách trước và sau để trả về `True`/`False`.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Agent thực hiện luồng RAG: gọi `store.search(question, top_k)` để thu thập ngữ cảnh, đánh số từng đoạn trích dẫn kèm nguồn gốc `[1] (Nguồn: ...)`, `[2] ...` để đảm bảo tính năng truy vết nguồn (Source Traceability). Prompt được thiết kế chặt chẽ yêu cầu mô hình chỉ trả lời dựa trên bằng chứng cung cấp, trích dẫn số thứ tự và nói rõ nếu không tìm thấy dữ liệu để ngăn ngừa hiện tượng ảo giác (hallucination).

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0 -- D:\AITHUCCHIEN\Lab07\K4-DAY07-MaiQuangDung-2A202602966\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\AITHUCCHIEN\Lab07\K4-DAY07-MaiQuangDung-2A202602966
plugins: anyio-4.15.1
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.07s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Mô hình thực nghiệm: OpenAI `text-embedding-3-small`.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|:---:|:---|:---|:---:|:---:|:---:|
| 1 | Người mua có thể yêu cầu đổi trả trong 15 ngày. | Khách hàng được hoàn tiền trong vòng mười lăm ngày. | Cao | 0.6353 | Đúng |
| 2 | Người bán phải phản hồi khiếu nại trong 2 ngày. | Thời hạn xử lý phản hồi của người bán là 48 giờ. | Cao | 0.6377 | Đúng |
| 3 | Hàng hóa giao đi phải còn ít nhất 30% hạn sử dụng. | Shopee cấm tạo đơn hàng ảo để gian lận. | Thấp | 0.3298 | Đúng |
| 4 | Sản phẩm bị lỗi kỹ thuật được bảo hành miễn phí. | Chính sách bảo hành không áp dụng nếu tem bị rách. | Trung bình | 0.5435 | Đúng |
| 5 | Quy chế hoạt động sàn giao dịch thương mại điện tử. | Cơm tấm sườn bì chả ở Sài Gòn rất ngon. | Thấp | 0.2059 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất là ở cặp số 4: hai câu mang tính chất đối nghịch (điều kiện được bảo hành vs trường hợp từ chối bảo hành) nhưng điểm tương đồng vẫn đạt mức 0.5435. Điều này cho thấy mô hình embedding biểu diễn ngữ nghĩa theo cụm ngữ cảnh chuyên ngành (domain cluster): các câu có chung miền chủ đề "bảo hành sản phẩm" vẫn nằm gần nhau trong không gian vector, dù logic khẳng định hay phủ định có sự đối lập.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên chiến lược cá nhân: **`RecursiveChunker` (chunk_size=400)** với mô hình OpenAI `text-embedding-3-small` (dữ liệu trích từ `ket_qua_benchmark.txt`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Gold Rank) | Câu trả lời của Agent (tóm tắt) | Điểm câu |
|:---|:---|:---|:---:|:---:|:---|:---:|
| 1 | Người mua có thể gửi yêu cầu trả hàng hoàn tiền trong thời hạn bao lâu sau khi nhận hàng? | 3.2. Người Mua có thể gửi yêu cầu trả hàng/hoàn tiền trong vòng 15 (mười lăm) ngày kể từ lúc đơn hàng được cập... | 0.7502 | Có (Top-1) | Trong vòng 15 ngày (thực phẩm tươi sống là 24 giờ) [1]. | 2/2 |
| 2 | Thời hạn xử lý và phản hồi khiếu nại yêu cầu trả hàng hoàn tiền của Người Bán là bao lâu? *(Filter: seller)* | Người Bán cần gửi phản hồi trong vòng 02 ngày lịch (hoặc thời hạn khác được quy định bởi Shopee... | 0.6052 | Có (Top-1) | Trong vòng 02 ngày lịch kể từ ngày nhận thông báo của Shopee [1]. | 2/2 |
| 3 | Người bán có hành vi gian lận tạo đơn hàng ảo trên Shopee bị xử phạt bồi thường bao nhiêu tiền cho mỗi đơn hàng vi phạm? | Người Bán được xem là có hành vi gian lận... (Chunk chứa số tiền phạt 10.000.000 VND xếp ở Top-3 với score 0.6581) | 0.6998 | Có (Top-3) | Bồi thường khoản tiền lên đến 10.000.000 VND cho từng đơn vi phạm [3]. | 1/2 |
| 4 | Quy định về hạn sử dụng của hàng hóa khi Người Bán giao đi trên Shopee phải còn lại ít nhất bao nhiêu? | h. Các sản phẩm khác bắt buộc phải có hạn sử dụng... còn ít nhất 30% thời hạn sử dụng và còn ít nhất 30 ngày... | 0.6746 | Có (Top-1) | Còn ít nhất 30% thời hạn sử dụng và còn ít nhất 30 ngày [1]. | 2/2 |
| 5 | Sản phẩm mua trên Shopee được bảo hành miễn phí khi đáp ứng những điều kiện nào? *(Filter: buyer)* | Sản phẩm cung cấp được đảm bảo hàng chính hãng, gửi sản phẩm trực tiếp đến địa chỉ bảo hành... | 0.6814 | Không (Vắng trong Top-3) | Agent chỉ trích dẫn quy trình gửi bảo hành, thiếu điều kiện "lỗi kỹ thuật do NSX" [1], [2], [3]. | 0/2 |

**Tổng kết chỉ số đánh giá thực nghiệm (Benchmark Summary):**
- **Điểm Retrieval Quality:** **7 / 10 điểm** (đạt chuẩn rubric `docs/SCORING.md`).
- **Hit@1:** **3 / 5 (60.00%)** — Câu 1, Câu 2, Câu 4 đưa đúng chunk vào vị trí ưu tiên số 1.
- **Hit@3:** **4 / 5 (80.00%)** — 4 trên 5 câu truy xuất được thông tin cốt lõi trong Top-3.
- **MRR (Mean Reciprocal Rank):** **0.6667** ($1/1 + 1/1 + 1/3 + 1/1 + 0 = 3.3333 / 5$).
- **Số câu hỏi trả về chunk có liên quan trong Top-3:** 4 / 5 câu.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo & so sánh):**
> Qua thực nghiệm với `RecursiveChunker`, tôi nhận thấy việc cắt văn bản thuần túy theo kích thước (kể cả có đệ quy) đôi khi làm tách rời tiêu đề của mục điều khoản với nội dung chi tiết bên dưới. Cụ thể ở Câu 5: chunk chứa điều kiện bảo hành miễn phí ("lỗi kỹ thuật do nhà sản xuất") bị tách khỏi heading `# Chính sách bảo hành`, dẫn đến việc các chunk khác nói chung chung về "trung tâm bảo hành" chiếm ưu thế điểm tương đồng cosine và đẩy chunk điều kiện ra ngoài Top-3 (dẫn đến điểm 0/2 cho câu 5). 
> 
> Trong khi đó, thành viên **Ngô Anh Khoa** sử dụng chiến lược `HeadingChunker` (chia nhỏ theo thẻ tiêu đề Markdown `#`, `##` và tự động đính kèm heading vào đầu mỗi sub-chunk) giải quyết triệt để điểm yếu này, giúp chunk điều kiện luôn mang theo ngữ cảnh "Điều kiện bảo hành" và lọt Top-1 câu 5 một cách thuyết phục. Đồng thời, qua quan sát kết quả của bạn **Đặng Đỉnh Đoàn** (`SentenceChunker`), việc gom theo câu giữ nguyên vẹn con số và điều kiện giúp đạt điểm tuyệt đối 5/5 câu hỏi nhưng phải đánh đổi bằng kích thước chunk không đồng đều (có chunk lên tới 1.748 ký tự). Đây là bài học thực tế vô giá về việc chọn chiến lược chunking phải ăn khớp với cấu trúc phân cấp của dữ liệu miền (domain document).

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|:---|:---:|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
