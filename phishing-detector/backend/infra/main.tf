resource "aws_s3_bucket" "model_storage" {
  bucket = "${var.project_name}-model-storage"
  tags = {
    Name    = "Phishing Detector Model Storage"
    Project = var.project_name
  }
}

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  tags = {
    Name = "${var.project_name}-vpc"
  }
}

resource "aws_ecr_repository" "phishing_api" {
  name                 = "${var.project_name}-api"
  image_tag_mutability = "MUTABLE"
  tags = {
    Name = "Phishing Detector API Repo"
  }
}

resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"
  tags = {
    Name = "Phishing Detector ECS Cluster"
  }
}

resource "aws_iam_role" "ecs_task_role" {
  name = "${var.project_name}-ecs-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_db_instance" "phishing_db" {
  identifier          = "${var.project_name}-db"
  engine              = "mysql"
  engine_version      = "8.0"
  instance_class      = "db.t3.micro"
  allocated_storage   = 20
  db_name             = "phishingdb"
  username            = "admin"
  password            = var.db_password
  skip_final_snapshot = true
  publicly_accessible = true
  tags = {
    Name    = "Phishing Detector RDS"
    Project = var.project_name
  }
}