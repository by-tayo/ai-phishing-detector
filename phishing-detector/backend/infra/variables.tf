variable "aws_region" {
  default = "us-east-1"
}

variable "project_name" {
  default = "phishing-detector"
}

variable "db_password" {
  description = "RDS MySQL password"
  sensitive   = true
}