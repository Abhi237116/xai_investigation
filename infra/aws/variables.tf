variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "enable_demo_bucket" {
  description = "Explicitly enable only the optional encrypted artifact bucket. No other resources are provisioned."
  type        = bool
  default     = false
}

variable "name_prefix" {
  type    = string
  default = "fci-demo"
}
