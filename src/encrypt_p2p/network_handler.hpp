#ifndef NETWORK_HANDLER_HPP
#define NETWORK_HANDLER_HPP

#include <iostream>
#include <string>
#include <zmq.hpp>
#include <cstring>
#include <vector>

namespace encrypt_p2p {
class NetworkHandler {
public:
    static zmq::context_t& getContext() {
        static zmq::context_t context(1);
        return context;
    }

    NetworkHandler(std::string local_ip, int local_port) : socket(getContext(), ZMQ_DEALER), connected(false), local_ip(local_ip), local_port(local_port) {}

    ~NetworkHandler() {
        disconnect();
    }

    // Connect to a remote endpoint.
    bool connect(const std::string &address, int port) {
        std::string endpoint = "tcp://" + address + ":" + std::to_string(port);
        try {
            socket.connect(endpoint);
            connected = true;
            endpoint_address = address;
            endpoint_port = port;
        } catch (const zmq::error_t &e) {
            std::cerr << "ZeroMQ connection error: " << e.what() << std::endl;
            connected = false;
        }
        return connected;
    }

    bool bind(const std::string &address, int port) {
        std::string endpoint = "tcp://" + address + ":" + std::to_string(port);
        try {
            socket.bind(endpoint);
            connected = true;
            endpoint_address = address;
            endpoint_port = port;
        } catch (const zmq::error_t &e) {
            std::cerr << "ZeroMQ binding error: " << e.what() << std::endl;
            connected = false;
        }
        return connected;
    }

    // Send data using this socket.
    bool sendData(const std::string &data) {
        if (!connected) {
            std::cerr << "Not connected. Cannot send data." << std::endl;
            return false;
        }
        // "\x01<local_ip>\x00<local_port>"
        std::string identity;
        identity.push_back('\x01');
        identity += local_ip;
        identity.push_back('\x00');
        identity += std::to_string(local_port);

        zmq::message_t id_msg(identity.size());
        memcpy(id_msg.data(), identity.data(), identity.size());
        
        // actual content.
        zmq::message_t content_msg(data.size());
        memcpy(content_msg.data(), data.data(), data.size());
        
        try {
            // Send the identity frame with the flag indicating more frames are coming.
            socket.send(id_msg, zmq::send_flags::sndmore);
            // Send the content frame as the final part.
            socket.send(content_msg, zmq::send_flags::none);
        } catch (const zmq::error_t &e) {
            std::cerr << "ZeroMQ send error: " << e.what() << std::endl;
            return false;
        }
        return true;
    } 

    std::string receiveData(std::string& sender_ip, int& sender_port, int timeout_ms = -1) {
        if (!connected) {
            std::cerr << "Not connected. Cannot receive data." << std::endl;
            return "";
        }
        
        zmq::pollitem_t item = { static_cast<void*>(socket), 0, ZMQ_POLLIN, 0 };
        int rc = zmq::poll(&item, 1, timeout_ms);
        if (rc == 0) {
            // Timeout: no data received.
            return "";
        }
        if (item.revents & ZMQ_POLLIN) {
            zmq::message_t identity;
            zmq::message_t content;
            try {
                socket.recv(identity, zmq::recv_flags::none);
                socket.recv(content, zmq::recv_flags::none);
                // format: \x01<ip>\x00<port>
                const unsigned char* data = static_cast<unsigned char*>(identity.data());
                std::string ip_port(reinterpret_cast<const char*>(data+1), identity.size()-1);
                size_t sep = ip_port.find('\0');
                
                sender_ip = ip_port.substr(0, sep);
                sender_port = std::stoi(ip_port.substr(sep+1));
            } catch (const zmq::error_t &e) {
                std::cerr << "ZeroMQ receive error: " << e.what() << std::endl;
                return "";
            }
            return std::string(static_cast<char*>(content.data()), content.size());
        }
        return "";
    }

    // Disconnect the socket.
    void disconnect() {
        if (connected) {
            socket.close();
            connected = false;
        }
    }

    bool isConnected() const {
        return connected;
    }

private:
    zmq::socket_t socket;
    bool connected;
    std::string endpoint_address;
    int endpoint_port;
    std::string local_ip;
    int local_port;
};
}
#endif 