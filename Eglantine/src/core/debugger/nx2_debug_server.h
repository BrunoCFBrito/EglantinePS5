// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "common/common_types.h"

/// Transport of the debug server (docs/nx2/debug-mcp.md): newline-delimited JSON requests over a
/// loopback TCP socket, served on one thread. It knows nothing about the emulator; the tools are
/// the handlers it is given (nx2_debug_tools.cpp).
namespace Core::Nx2Debug {

struct Endpoint {
    std::string host;
    u16 port;
};

/// Parses EDEN_NX2_DEBUG_SERVER / EDEN_NX2_DEBUG_CONNECT: "<port>" or "<host>:<port>". Only
/// loopback hosts are accepted unless EDEN_NX2_DEBUG_ALLOW_REMOTE=1, or `any_host` is set for a
/// connection this process makes (dialling out exposes nothing).
[[nodiscard]] std::optional<Endpoint> ParseEndpoint(std::string_view text, std::string& error,
                                                    bool any_host = false);

/// Whether the client of the request being handled has disconnected. A handler that waits polls
/// it to undo what it started (input injection releases its buttons).
using ClientGone = std::function<bool()>;

/// A tool: takes the request's params, returns its result, throws std::exception on failure.
using Handler =
    std::function<nlohmann::json(const nlohmann::json& params, const ClientGone& client_gone)>;
using Handlers = std::map<std::string, Handler, std::less<>>;

/// Answers one request line, {"id":..,"method":..,"params":{..}}, with one response line (no
/// newline): {"id":..,"result":..} or {"id":..,"error":"..."}.
[[nodiscard]] std::string HandleRequest(const Handlers& handlers, std::string_view line,
                                        const ClientGone& client_gone = [] { return false; });

[[nodiscard]] std::string Base64(std::span<const u8> data);

class Server {
public:
    /// Listens on the endpoint (port 0 picks a free one); throws std::exception when it cannot.
    explicit Server(const Endpoint& endpoint, Handlers handlers);
    ~Server();

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    [[nodiscard]] u16 Port() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

/// Dials <host>:<port> and serves the same requests over that connection, dialling again a second
/// after it is lost. A PS5 title's accept() fails with EACCES (verified on the console), so there
/// the game connects out to the machine running the client instead
/// (EDEN_NX2_DEBUG_CONNECT=<host>:<port>).
class Client {
public:
    explicit Client(const Endpoint& endpoint, Handlers handlers);
    ~Client();

    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

/// Serves the same requests through two files in a directory the client reaches over FTP:
/// request.json holds one request line, response.json the answer, both tagged with "seq". A PS5
/// title is refused TCP in both directions (accept() and connect() return EACCES, verified
/// 2026-11-19), and the filesystem is what is left. One request per poll (EDEN_NX2_DEBUG_POLL_MS,
/// 200 ms by default).
class FileSession {
public:
    explicit FileSession(std::string directory, Handlers handlers);
    ~FileSession();

    FileSession(const FileSession&) = delete;
    FileSession& operator=(const FileSession&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Core::Nx2Debug
