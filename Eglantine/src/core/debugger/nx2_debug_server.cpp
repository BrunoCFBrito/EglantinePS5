// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <thread>

#include <boost/asio.hpp>

#include "common/logging.h"
#include "common/thread.h"
#include "core/debugger/nx2_debug_server.h"

namespace Core::Nx2Debug {
namespace {

using boost::asio::ip::tcp;

/// Longest request line; a client sending more is disconnected.
constexpr size_t MAX_REQUEST = size_t{1} << 20;

class Session : public std::enable_shared_from_this<Session> {
public:
    explicit Session(tcp::socket socket_, const Handlers& handlers_,
                     std::function<void()> on_close_ = {})
        : socket{std::move(socket_)}, handlers{handlers_}, on_close{std::move(on_close_)} {}

    void Read() {
        boost::asio::async_read_until(
            socket, boost::asio::dynamic_buffer(input, MAX_REQUEST), '\n',
            [self = shared_from_this()](const boost::system::error_code& error, size_t length) {
                if (error) {
                    self->Closed();
                    return;
                }
                std::string_view line{self->input.data(), length - 1};
                if (!line.empty() && line.back() == '\r') {
                    line.remove_suffix(1);
                }
                self->output = line.empty() ? std::string{}
                                            : HandleRequest(self->handlers, line,
                                                            [&self] { return self->ClientGone(); });
                self->input.erase(0, length);
                if (self->output.empty()) {
                    self->Read();
                    return;
                }
                self->output += '\n';
                self->Write();
            });
    }

private:
    /// Tells the owner once that the connection is over, so a dialling client can connect again.
    void Closed() {
        if (on_close) {
            auto callback = std::move(on_close);
            on_close = nullptr;
            callback();
        }
    }

    /// A non-blocking peek: end of stream or a reset means the client left.
    bool ClientGone() {
        boost::system::error_code error;
        socket.non_blocking(true, error);
        std::array<char, 1> byte;
        const size_t length = socket.receive(boost::asio::buffer(byte),
                                             tcp::socket::message_peek, error);
        return error ? error != boost::asio::error::would_block : length == 0;
    }

    void Write() {
        boost::asio::async_write(
            socket, boost::asio::buffer(output),
            [self = shared_from_this()](const boost::system::error_code& error, size_t) {
                if (error) {
                    self->Closed();
                    return;
                }
                self->Read();
            });
    }

    tcp::socket socket;
    const Handlers& handlers;
    std::function<void()> on_close;
    std::string input;
    std::string output;
};

} // Anonymous namespace

std::optional<Endpoint> ParseEndpoint(std::string_view text, std::string& error, bool any_host) {
    std::string_view host = "127.0.0.1";
    std::string_view port_text = text;
    if (const size_t colon = text.rfind(':'); colon != std::string_view::npos) {
        host = text.substr(0, colon);
        port_text = text.substr(colon + 1);
        if (host.size() >= 2 && host.front() == '[' && host.back() == ']') {
            host = host.substr(1, host.size() - 2);
        }
    }
    if (host == "localhost") {
        host = "127.0.0.1";
    }
    // EDEN_NX2_DEBUG_ALLOW_REMOTE=1 lets the console's server bind a LAN address so a build
    // machine can query it. Off by default: the protocol has no authentication.
    static const bool allow_remote = [] {
        const char* const value = std::getenv("EDEN_NX2_DEBUG_ALLOW_REMOTE");
        return value != nullptr && value[0] == '1';
    }();
    if (!any_host && !allow_remote && host != "127.0.0.1" && host != "::1") {
        error = fmt::format("host {} is not loopback; only 127.0.0.1, ::1 and localhost are allowed "
                            "(EDEN_NX2_DEBUG_ALLOW_REMOTE=1 permits any address)",
                            host);
        return std::nullopt;
    }
    unsigned port = 0;
    const auto [end, result] =
        std::from_chars(port_text.data(), port_text.data() + port_text.size(), port);
    if (port_text.empty() || result != std::errc{} || end != port_text.data() + port_text.size() ||
        port > 65535) {
        error = fmt::format("bad port \"{}\"", port_text);
        return std::nullopt;
    }
    return Endpoint{std::string{host}, static_cast<u16>(port)};
}

std::string HandleRequest(const Handlers& handlers, std::string_view line,
                          const ClientGone& client_gone) {
    nlohmann::json response = {{"id", nullptr}};
    try {
        const nlohmann::json request = nlohmann::json::parse(line);
        if (!request.is_object()) {
            throw std::runtime_error{"request is not an object"};
        }
        response["id"] = request.value("id", nlohmann::json{});
        const std::string method = request.value("method", std::string{});
        const auto it = handlers.find(method);
        if (it == handlers.end()) {
            throw std::runtime_error{fmt::format("unknown method \"{}\"", method)};
        }
        const nlohmann::json params = request.value("params", nlohmann::json::object());
        if (!params.is_object()) {
            throw std::runtime_error{"params is not an object"};
        }
        response["result"] = it->second(params, client_gone);
    } catch (const std::exception& error) {
        response = {{"id", response["id"]}, {"error", error.what()}};
    }
    return response.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::string Base64(std::span<const u8> data) {
    static constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((data.size() + 2) / 3 * 4);
    for (size_t index = 0; index < data.size(); index += 3) {
        const size_t left = data.size() - index;
        const u32 chunk = (u32{data[index]} << 16) | (left > 1 ? u32{data[index + 1]} << 8 : 0) |
                          (left > 2 ? u32{data[index + 2]} : 0);
        out += alphabet[(chunk >> 18) & 63];
        out += alphabet[(chunk >> 12) & 63];
        out += left > 1 ? alphabet[(chunk >> 6) & 63] : '=';
        out += left > 2 ? alphabet[chunk & 63] : '=';
    }
    return out;
}

struct Server::Impl {
    explicit Impl(const Endpoint& endpoint, Handlers handlers_)
        : handlers{std::move(handlers_)}, acceptor{io} {
        const tcp::endpoint address{boost::asio::ip::make_address(endpoint.host), endpoint.port};
        acceptor.open(address.protocol());
#ifndef _WIN32
        // Rebinding after a relaunch while the old socket is in TIME_WAIT. On Windows the option
        // would let a second instance bind the same port instead.
        acceptor.set_option(tcp::acceptor::reuse_address(true));
#endif
        acceptor.bind(address);
        acceptor.listen();
        Accept();
        thread = std::thread{[this] {
            Common::SetCurrentThreadName("DebugServer");
            io.run();
        }};
    }

    ~Impl() {
        io.stop();
        if (thread.joinable()) {
            thread.join();
        }
    }

    void Accept() {
        acceptor.async_accept([this](const boost::system::error_code& error, tcp::socket socket) {
            if (error) {
                // Re-arming only on success would end the accept loop for good on the first
                // transient failure, leaving the port open with nothing behind it.
                if (error == boost::asio::error::operation_aborted) {
                    return;
                }
                LOG_WARNING(Debug, "Debug server accept failed: {}", error.message());
                retry_timer.expires_after(std::chrono::seconds{1});
                retry_timer.async_wait([this](const boost::system::error_code&) { Accept(); });
                return;
            }
            LOG_INFO(Debug, "Debug server: client connected");
            std::make_shared<Session>(std::move(socket), handlers)->Read();
            Accept();
        });
    }

    u16 Port() const noexcept {
        boost::system::error_code error;
        return acceptor.local_endpoint(error).port();
    }

    Handlers handlers;
    boost::asio::io_context io;
    tcp::acceptor acceptor;
    boost::asio::steady_timer retry_timer{io};
    // Keeps io.run() alive while only the timer or no handler is pending.
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type> work{
        boost::asio::make_work_guard(io)};
    std::thread thread;
};

Server::Server(const Endpoint& endpoint, Handlers handlers)
    : impl{std::make_unique<Impl>(endpoint, std::move(handlers))} {}

Server::~Server() = default;

u16 Server::Port() const noexcept {
    return impl->Port();
}

struct Client::Impl {
    Impl(const Endpoint& endpoint, Handlers handlers_)
        : host{endpoint.host}, port{endpoint.port}, handlers{std::move(handlers_)}, socket{io} {
        Connect();
        thread = std::thread{[this] {
            Common::SetCurrentThreadName("DebugClient");
            io.run();
        }};
    }

    ~Impl() {
        io.stop();
        if (thread.joinable()) {
            thread.join();
        }
    }

    void Connect() {
        // A socket is spent by the attempt (or the session) it was made for.
        socket = tcp::socket{io};
        // A console's getaddrinfo (ps5_getaddrinfo) refuses a numeric address, so an address is
        // connected to directly and only a name goes to the resolver.
        boost::system::error_code parsed;
        const auto address = boost::asio::ip::make_address(host, parsed);
        if (!parsed) {
            socket.async_connect(tcp::endpoint{address, port},
                                 [this](const boost::system::error_code& error) {
                                     Connected(error);
                                 });
            return;
        }
        resolver.async_resolve(
            host, std::to_string(port),
            [this](const boost::system::error_code& error, tcp::resolver::results_type results) {
                if (error) {
                    Retry(error);
                    return;
                }
                boost::asio::async_connect(
                    socket, results, [this](const boost::system::error_code& error,
                                            const tcp::endpoint&) { Connected(error); });
            });
    }

    void Connected(const boost::system::error_code& error) {
        if (error) {
            Retry(error);
            return;
        }
        LOG_INFO(Debug, "Debug client connected to {}:{}", host, port);
        failures = 0;
        std::make_shared<Session>(std::move(socket), handlers, [this] { Retry({}); })->Read();
    }

    void Retry(const boost::system::error_code& error) {
        if (error && (failures++ % 60) == 0) {
            LOG_INFO(Debug, "Debug client cannot reach {}:{}: {}; retrying", host, port,
                     error.message());
        }
        retry_timer.expires_after(std::chrono::seconds{1});
        retry_timer.async_wait([this](const boost::system::error_code& cancelled) {
            if (!cancelled) {
                Connect();
            }
        });
    }

    std::string host;
    u16 port;
    Handlers handlers;
    boost::asio::io_context io;
    tcp::resolver resolver{io};
    boost::asio::steady_timer retry_timer{io};
    tcp::socket socket;
    unsigned failures{};
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type> work{
        boost::asio::make_work_guard(io)};
    std::thread thread;
};

Client::Client(const Endpoint& endpoint, Handlers handlers)
    : impl{std::make_unique<Impl>(endpoint, std::move(handlers))} {}

Client::~Client() = default;

struct FileSession::Impl {
    Impl(std::string directory, Handlers handlers_)
        : dir{std::filesystem::path{directory} / "request.json"},
          answer{std::filesystem::path{directory} / "response.json"},
          handlers{std::move(handlers_)} {
        std::error_code error;
        std::filesystem::create_directories(dir.parent_path(), error);
        thread = std::thread{[this] {
            Common::SetCurrentThreadName("DebugFile");
            Serve();
        }};
    }

    ~Impl() {
        stopping.store(true, std::memory_order_relaxed);
        if (thread.joinable()) {
            thread.join();
        }
    }

    /// Reads the whole file, or an empty string while the client is still writing it.
    static std::string ReadAll(const std::filesystem::path& path) {
        std::ifstream file{path, std::ios::binary};
        return file ? std::string{std::istreambuf_iterator<char>{file}, {}} : std::string{};
    }

    void Serve() {
        unsigned interval_ms = 200;
        if (const char* const text = std::getenv("EDEN_NX2_DEBUG_POLL_MS")) {
            interval_ms = static_cast<unsigned>(std::clamp(std::strtoul(text, nullptr, 10), 20ul, 5000ul));
        }
        u64 last_seq = 0;
        while (!stopping.load(std::memory_order_relaxed)) {
            const std::string request = ReadAll(dir);
            if (!request.empty()) {
                u64 seq = 0;
                try {
                    const nlohmann::json parsed = nlohmann::json::parse(request);
                    seq = parsed.value("seq", u64{0});
                } catch (const std::exception&) {
                    seq = 0;  // still being uploaded; try again on the next poll
                }
                if (seq != 0 && seq != last_seq) {
                    last_seq = seq;
                    const std::string response = HandleRequest(handlers, request);
                    const auto temporary = answer.string() + ".new";
                    {
                        std::ofstream file{temporary, std::ios::binary | std::ios::trunc};
                        if (file) {
                            file << response << '\n';
                        }
                    }
                    std::error_code error;
                    std::filesystem::rename(temporary, answer, error);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{interval_ms});
        }
    }

    std::filesystem::path dir;
    std::filesystem::path answer;
    Handlers handlers;
    std::atomic<bool> stopping{false};
    std::thread thread;
};

FileSession::FileSession(std::string directory, Handlers handlers)
    : impl{std::make_unique<Impl>(std::move(directory), std::move(handlers))} {}

FileSession::~FileSession() = default;

} // namespace Core::Nx2Debug
