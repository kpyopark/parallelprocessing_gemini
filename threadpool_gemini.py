from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from vertexai.language_models import TextGenerationModel
import vertexai
import time
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiParallelProcessor:
    def __init__(
        self,
        project_id: str,
        location: str,
        input_query: str,
        output_table: str,
        result_table: str,
        batch_size: int = 100000,
        process_size: int = 10000,
        max_workers: int = 10
    ):
        self.project_id = project_id
        self.location = location
        self.input_query = input_query
        self.output_table = output_table
        self.result_table = result_table
        self.batch_size = batch_size
        self.process_size = process_size
        self.max_workers = max_workers
        
        # BigQuery 클라이언트 초기화
        self.bq_client = bigquery.Client(project=project_id)
        
        # Vertex AI 초기화
        vertexai.init(project=project_id, location=location)
        self.model = TextGenerationModel.from_pretrained("gemini-pro")
        
        # 결과 테이블 스키마 생성
        self.create_result_table()

    def create_result_table(self):
        """결과 저장을 위한 테이블 생성"""
        schema = [
            bigquery.SchemaField("batch_id", "STRING"),
            bigquery.SchemaField("process_start_time", "TIMESTAMP"),
            bigquery.SchemaField("process_end_time", "TIMESTAMP"),
            bigquery.SchemaField("processed_count", "INTEGER"),
            bigquery.SchemaField("success_count", "INTEGER"),
            bigquery.SchemaField("error_count", "INTEGER"),
            bigquery.SchemaField("status", "STRING")
        ]
        
        table = bigquery.Table(f"{self.project_id}.{self.result_table}", schema=schema)
        try:
            self.bq_client.create_table(table)
        except Exception as e:
            logger.info(f"Result table already exists or error: {str(e)}")

    def process_single_request(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """단일 Gemini API 요청 처리"""
        try:
            # 여기에 실제 Gemini API 호출 로직 구현
            # 예시: 입력 텍스트를 기반으로 응답 생성
            response = self.model.predict(
                row['input_text'],
                temperature=0.7,
                max_output_tokens=1024
            )
            
            return {
                'id': row['id'],
                'input_text': row['input_text'],
                'output_text': response.text,
                'status': 'success',
                'error': None,
                'processed_at': datetime.now()
            }
        except Exception as e:
            return {
                'id': row['id'],
                'input_text': row['input_text'],
                'output_text': None,
                'status': 'error',
                'error': str(e),
                'processed_at': datetime.now()
            }

    def save_results_to_bq(self, results: List[Dict[str, Any]], batch_id: str):
        """처리된 결과를 BigQuery에 저장"""
        df = pd.DataFrame(results)
        
        # 결과를 BigQuery 테이블에 저장
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND"
        )
        
        self.bq_client.load_table_from_dataframe(
            df,
            f"{self.project_id}.{self.output_table}",
            job_config=job_config
        )

    def save_batch_status(self, batch_id: str, start_time: datetime, 
                         end_time: datetime, processed_count: int,
                         success_count: int, error_count: int):
        """배치 처리 상태를 결과 테이블에 저장"""
        query = f"""
        INSERT INTO `{self.project_id}.{self.result_table}`
        (batch_id, process_start_time, process_end_time, processed_count, 
         success_count, error_count, status)
        VALUES
        ('{batch_id}', 
         TIMESTAMP('{start_time.isoformat()}'), 
         TIMESTAMP('{end_time.isoformat()}'),
         {processed_count}, {success_count}, {error_count},
         'COMPLETED')
        """
        self.bq_client.query(query).result()

    def process_batch(self):
        """배치 단위로 데이터 처리"""
        offset = 0
        while True:
            # 배치 크기만큼 데이터 조회
            query = f"""
            {self.input_query}
            LIMIT {self.batch_size}
            OFFSET {offset}
            """
            
            df = self.bq_client.query(query).to_dataframe()
            
            if df.empty:
                break
                
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{offset}"
            start_time = datetime.now()
            
            # 처리 크기 단위로 나누어 처리
            for i in range(0, len(df), self.process_size):
                process_chunk = df[i:i + self.process_size].to_dict('records')
                results = []
                success_count = 0
                error_count = 0
                
                # ThreadPoolExecutor를 사용한 병렬 처리
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_row = {
                        executor.submit(self.process_single_request, row): row 
                        for row in process_chunk
                    }
                    
                    for future in as_completed(future_to_row):
                        result = future.result()
                        results.append(result)
                        if result['status'] == 'success':
                            success_count += 1
                        else:
                            error_count += 1
                
                # 결과를 BigQuery에 저장
                self.save_results_to_bq(results, batch_id)
                
                # 진행 상황 로깅
                logger.info(f"Processed {len(results)} items in batch {batch_id}")
            
            # 배치 처리 완료 상태 저장
            end_time = datetime.now()
            self.save_batch_status(
                batch_id, start_time, end_time,
                len(df), success_count, error_count
            )
            
            offset += self.batch_size

# 사용 예시
if __name__ == "__main__":
    processor = GeminiParallelProcessor(
        project_id="your-project-id",
        location="us-central1",
        input_query="""
        SELECT 
            id,
            input_text
        FROM `your-project.your-dataset.your-input-table`
        WHERE processed_flag = False
        ORDER BY id
        """,
        output_table="your-dataset.processing_results",
        result_table="your-dataset.batch_status",
        batch_size=100000,
        process_size=10000,
        max_workers=10
    )
    
    processor.process_batch()