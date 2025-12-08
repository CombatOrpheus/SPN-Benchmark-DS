provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_storage_bucket" "data_bucket" {
  name          = var.bucket_name
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true
}

resource "google_cloud_run_v2_job" "default" {
  name     = var.job_name
  location = var.region

  template {
    template {
      containers {
        image = var.image_name

        env {
          name  = "GCS_BUCKET_NAME"
          value = google_storage_bucket.data_bucket.name
        }

        # SPN_CONFIG_JSON is expected to be overridden at execution time.
        # We set a minimal valid JSON default to avoid crash if run without args,
        # though the script will likely fail generation if config is empty but valid JSON.
        env {
          name = "SPN_CONFIG_JSON"
          value = "{}"
        }

        resources {
          limits = {
            cpu    = "2"
            memory = "4Gi"
          }
        }
      }

      # Set timeout to 30 minutes (1800 seconds)
      timeout = "1800s"
    }
  }
}
