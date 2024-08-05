#include <stdio.h>
#include <stdlib.h>
#include <curl/curl.h>

// Callback function to write data to a file
size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t written = fwrite(ptr, size, nmemb, stream);
    return written;
}

int main() {
    CURL *curl;
    FILE *fp;
    CURLcode res;
    const char *url = "http://169.254.68.156/cgi-bin/viewer/video.jpg";
    const char *outfilename = "capture.jpg";
    const char *username = "admin";
    const char *password = "admin";

    // Initialize curl
    curl = curl_easy_init();
    if (curl) {
        // Open file to write image
        fp = fopen(outfilename, "wb");
        if (!fp) {
            perror("Error opening file");
            return 1;
        }

        // Set curl options
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

        // Set HTTP authentication
        curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
        curl_easy_setopt(curl, CURLOPT_USERNAME, username);
        curl_easy_setopt(curl, CURLOPT_PASSWORD, password);

        // Perform the request
        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        }

        // Clean up
        fclose(fp);
        curl_easy_cleanup(curl);
    } else {
        fprintf(stderr, "Error initializing curl\n");
        return 1;
    }

    return 0;
}