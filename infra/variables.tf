variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "image_name" {
  description = "Container image URI"
  type        = string
}

variable "job_name" {
  description = "Cloud Run Job name"
  type        = string
  default     = "spn-generator-job"
}

variable "bucket_name" {
    description = "Name of the GCS bucket to create"
    type        = string
}
