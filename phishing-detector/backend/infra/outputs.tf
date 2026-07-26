output "s3_bucket_name" {
  value = aws_s3_bucket.model_storage.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.phishing_api.repository_url
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "rds_endpoint" {
  value = aws_db_instance.phishing_db.endpoint
}