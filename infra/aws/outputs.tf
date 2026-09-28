output "artifact_bucket_name" {
  value       = var.enable_demo_bucket ? aws_s3_bucket.artifacts[0].bucket : null
  description = "Null by default; opt-in bucket only."
}
