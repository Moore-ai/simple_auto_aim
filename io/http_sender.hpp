#ifndef IO__HTTP_SENDER_HPP
#define IO__HTTP_SENDER_HPP

#include <Eigen/Dense>

#include <string>

typedef void CURL;

namespace io
{
class HttpSender
{
public:
  explicit HttpSender(const std::string & config_path);
  HttpSender(std::string url, long timeout_ms);
  ~HttpSender();

  HttpSender(const HttpSender &) = delete;
  HttpSender & operator=(const HttpSender &) = delete;

  bool send(const Eigen::Vector4d & target_info);

private:
  std::string url_;
  long timeout_ms_;
  CURL * curl_;
};
}  // namespace io

#endif  // IO__HTTP_SENDER_HPP
