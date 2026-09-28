resource "aws_s3_bucket" "artifacts" {
  count         = var.enable_demo_bucket ? 1 : 0
  bucket_prefix = "${var.name_prefix}-artifacts-"
  force_destroy = false
  tags = {
    Project   = "financial-crime-investigation"
    ManagedBy = "terraform"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  count  = var.enable_demo_bucket ? 1 : 0
  bucket = aws_s3_bucket.artifacts[0].id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  count                   = var.enable_demo_bucket ? 1 : 0
  bucket                  = aws_s3_bucket.artifacts[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
