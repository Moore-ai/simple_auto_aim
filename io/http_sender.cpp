#include "http_sender.hpp"

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <utility>

#include "tools/yaml.hpp"

namespace io
{
namespace
{
size_t discard_response(char * data, size_t size, size_t count, void * user_data)
{
  (void)data;
  (void)user_data;
  return size * count;
}
}  // namespace

HttpSender::HttpSender(const std::string & config_path)
: HttpSender(
    tools::read<std::string>(tools::load(config_path)["http_sender"], "url"),
    tools::read<long>(tools::load(config_path)["http_sender"], "timeout_ms"))
{
}

HttpSender::HttpSender(std::string url, long timeout_ms)
: url_(std::move(url)), timeout_ms_(timeout_ms), curl_(curl_easy_init())
{
}

HttpSender::~HttpSender()
{
  if (curl_ != nullptr) curl_easy_cleanup(curl_);
}

bool HttpSender::send(const Eigen::Vector4d & target_info)
{
  if (curl_ == nullptr) return false;

  const nlohmann::json body = {
    {"x", target_info[0]},
    {"y", target_info[1]},
    {"detected", target_info[2] != 0.0},
    {"target_id", static_cast<int>(target_info[3])}};
  const std::string payload = body.dump();

  curl_slist * headers = curl_slist_append(nullptr, "Content-Type: application/json");
  curl_easy_setopt(curl_, CURLOPT_URL, url_.c_str());
  curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, payload.c_str());
  curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, static_cast<long>(payload.size()));
  curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT_MS, timeout_ms_);
  curl_easy_setopt(curl_, CURLOPT_TIMEOUT_MS, timeout_ms_);
  curl_easy_setopt(curl_, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl_, CURLOPT_NOPROXY, "*");
  curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, discard_response);

  const CURLcode result = curl_easy_perform(curl_);
  long status_code = 0;
  if (result == CURLE_OK) curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &status_code);
  curl_slist_free_all(headers);
  return result == CURLE_OK && status_code >= 200 && status_code < 300;
}
}  // namespace io
