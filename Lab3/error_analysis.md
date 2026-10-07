Query: `soccer game`

# Ba similarity đúng

## Document 1915: 
Observed: Document được xếp ở rank 4 về similarity với query, đề cập đến quảng cáo trận đấu bóng đá.

Expected: Document đề cập đến trận đấu bóng đá.

Assessment: Đúng

Evidence from corpus: Document chứa các từ như football, game, championship, team trong corpus

Possible explanation:
Similarity cao vì document có các từ liên quan đến trận đấu bóng đá.

## Document 16031:
Observed: Document được xếp ở rank 5 về similarity với query, đề cập đến quảng cáo game về bóng đá.

Expected: Document đề cập đến trận đấu bóng đá.

Assessment: Đúng

Evidence from corpus: Document chứa các từ như goalkeeper (thủ môn), games, team, victory trong corpus

Possible explanation:
Similarity cao vì document có các từ liên quan đến chuyên ngành về bóng đá.

## Document 20399
Observed: Document được xếp ở rank 8 về similarity với query, đề cập đến quảng cáo website cập nhật tỷ số bóng đá.

Expected: Document đề cập đến trận đấu bóng đá.

Assessment: Đúng

Evidence from corpus: Document chứa các từ như results, tournaments, football, season, scores trong corpus

Possible explanation:
Similarity cao vì document có các từ liên quan đến bóng đá.

# Ba similarity sai

## Document 24068

Observed: Document được xếp ở rank 1 về similarity với query, đề cập đến quảng cáo website cập nhật kết quả tennis.

Expected: Document đề cập đến trận đấu bóng đá.

Assessment: Sai

Evidence from corpus: Document chứa từ game, results nhưng trong ngữ cảnh về tennis

Possible explanation:
Similarity cao vì có xuất hiện các từ như scores, results hay một số từ liên quan đến thể thao.

## Document 23546

Observed: Document được xếp ở rank 2 về similarity với query, đề cập đến lời mời chơi game không liên quan đến bóng đá.

Expected: Document đề cập đến trận đấu bóng đá.

Assessment: Sai

Evidence from corpus: Document đề cập đến các từ như 'league', 'play' nhưng liên quan đến game không liên quan đến bóng đá.

Possible explanation:
Document vector cũng được tạo bằng trung bình các word vector, nên similarity gần với query.

## Document 25572

Observed: Document được xếp ở rank 3 về similarity với query, đề cập đến quảng cáo trang mạng xã hội về bóng chày.

Expected: Document đề cập đến trận đấu bóng đá.

Assessment: Sai

Evidence from corpus: Document đề cập đến ngữ cảnh về bóng chày.

Possible explanation:
Có sự xuất hiện của các từ như 'league', 'player'