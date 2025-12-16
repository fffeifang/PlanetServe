#ifndef USER_NODE_HPP
#define USER_NODE_HPP

#include <string>
#include "../encrypt_p2p/network_handler.hpp"
#include "../encrypt_p2p/key_generation.hpp"
#include "../encrypt_p2p/crypto_utils.hpp"
#include "../encrypt_p2p/s_ida.hpp"
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <queue>
#include <map>

#define MODEL_NODE_PORT 8080
#define SIDA_DEFAULT_N 4  // Default number of paths for S-IDA
#define SIDA_DEFAULT_K 3  // Default threshold for S-IDA

/* sessionID should always be plain text*/
/* seq_num should not be plain text*/
/* should  is_last_message be plain text? */
struct Message{
    std::string sessionID;
    int seq_num;
    std::string question;
    std::vector<std::string> proxy_list;
    std::string model_IP_address;
    bool is_last_message;
};

namespace node {



class UserNode {
    private:
        std::string IP_ADDRESS;
        int PORT; 
        std::string RSA_PUBLIC_KEY;
        std::string RSA_PRIVATE_KEY;
        std::string IP_verification_node; // IP address of the verification node this node is registered to (a close one)
        std::vector<std::tuple<std::string,int, std::string>> UserNodesList; // <IP_ADDRESS, port, RSA_PUBLIC_KEY> for all nodes in same region
        std::vector<std::tuple<std::string,int, std::string>> ModelNodeList; // <IP_ADDRESS, port, RSA_PUBLIC_KEY> of model node list (reputation score > threshold), public key of model node for verify signature (actually only use for verification node).
        std::vector<std::string> ConversationList; // list of sessionID started by this node
        std::vector<std::tuple<std::string,std::string, bool>> ConversationList_proxy; // list of <sessionID, model_IP_address, ack> this nodeact as a proxy node (ack determines if the model_IP_address is actual)
        // martrix for received message?? no need for this now
        std::vector<std::vector<std::string>> ReceivedMessage; // first dimension is sessionID, second dimension is message sequence number
        std::vector<std::tuple<std::string, std::string, int, std::string, int>> RelayStateTable; // <sessionID, predecessor_node_IP, predecessor_node_port, successor_node_IP, successor_node_port> from start node to proxy node
        std::vector<std::string> VerificationNodeList; // IP_ADDRESS 
        
        std::vector<std::string> allQuestions; 
        std::vector<std::string> ActiveSessionTable; // sessionID started by this node and not ended
        std::map<std::string, encrypt_p2p::NetworkHandler> connectionPool; // for reuse
        encrypt_p2p::NetworkHandler listener; // for listening
        std::thread workerThread;
        std::atomic<bool> running{false};
        std::condition_variable cv;
        std::mutex mtx;
        std::mutex connectionMtx; // mutex for connectionPool
        std::queue<std::tuple<std::string, int, std::string>> messageQueue; // <sender_IP, sender_port, message>
        std::thread receiverThread;
        std::thread processorThread;
        
        // S-IDA message cache for reconstruction
        std::map<std::string, std::map<int, std::vector<encrypt_p2p::SIDA::Clove>>> sidaMessageCache; // sessionID -> seq_num -> cloves
        std::mutex sidaCacheMtx; // Mutex for SIDA message cache

    public:
        std::vector<std::tuple<std::string, std::pair<std::string,int>, std::pair<std::string,int>, std::pair<std::string,int>, std::pair<std::string,int>>> ProxyNodeList; // <sessionID, <Proxy_IP_ADDRESS, Proxy_port>, <Proxy_IP_ADDRESS, Proxy_port>, <Proxy_IP_ADDRESS, Proxy_port>, <Proxy_IP_ADDRESS, Proxy_port>> of proxy node list (proxy nodes of this node)
        std::vector<std::pair<std::string, std::vector<std::vector<std::pair<std::string, int>>>>> proxy_IP_path; //TODO: may need change format
        UserNode(std::string ip_address, int port);
        ~UserNode();
        // Initialize the node (e.g. setup resources)
        void initialize();
        std::string httpPost(const std::string &url, const std::string &jsonPayload);
        // Get the list of nodes in the same region from verifcation nore of ID of 
        void getNodesList(); // TODO: may distributed by verification node

        // Connection pool management
        bool SendWithConnection(const std::string& target_ip, int target_port, const std::string& data);
        void CleanupIdleConnections();
        std::string GenerateSessionID(); // generate a new sessionID, unique accross all nodes
        std::vector<Message> CreateSession(int N); // create a new session, TODO: confirm datasets
        bool EstablishProxyConnection(std::string &sessionID, std::vector<std::vector<std::pair<std::string, int>>> &proxy_IP_path, std::string &model_IP_address, int model_port); // try to establish N proxy connections
        std::string SendSession(std::vector<Message> &session, std::string model_IP_address, int n=SIDA_DEFAULT_N, int k=SIDA_DEFAULT_K);
        // get actual model node IP address
        std::string ReceiveProbeMessage(std::string &sessionID, int seq_num); // receive answer of message + actual model node IP address
        std::string ReceiveRegularMessage(std::string &sessionID, int seq_num, int n=SIDA_DEFAULT_N, int k=SIDA_DEFAULT_K); // receive message from final model node (after receiving probe message)
        void StartHandleMessage(); // continuiously listen to messages from other nodes
        void StopHandleMessage(); // stop listening to messages from other nodes

        void receiveLoop(); // receive messages from other nodes and put them into the message queue
        void processMessages(); // process the messages in the message queue

        /* 
        format of  forward regular message:
        R|sessionID encrypted_message(contains seq_num)
        */
        bool ForwardRegularMessage(std::string &sessionID, std::string &next_hop_ip, int next_hop_port, std::string &encrypted_message); // forward message to the next hop after establishing proxy connection
        /* 
        format of  forward probe message:
        P|sessionID (next_hop_IP next_hop_port encrypted_message)
        */
        bool ForwardProbeMessage(std::string &predecessor_node_IP, int predecessor_node_port, std::string &sessionID, std::string &msg); // forward message to the next hop
        /* 
        format of  receive probe message:
        H|sessionID (model_IP_address)
        */
        bool HandleProbeMessage(std::string &predecessor_node_IP, int predecessor_node_port, std::string &sessionID, std::string &model_IP_address); // act as a proxy node for probe message
        /*
        format of  receive confirmation message:
        C|sessionID 
        */
        bool ForwardConfirmationMessage(std::string &next_hop_ip, int next_hop_port, std::string &sessionID); // forward confirmation message to the next hop
        /*
        format of  receive confirmation message:
        C|sessionID 
        */
        bool HandleConfirmationMessage(std::string &sender_node_IP, int sender_node_port, std::string &sessionID); // act as a proxy node for confirmation message

    /*
        Methods for S-IDA message handling
        */
        bool SendSIDAMessage(const std::string& sessionID, int seq_num, const std::string& message, 
                             const std::vector<std::pair<std::string, int>>& paths, int n, int k);
        
        bool ProcessSIDAClove(const std::string& sessionID, int seq_num, const std::string& serializedClove);
        
        std::string ReconstructSIDAMessage(const std::string& sessionID, int seq_num, int k);

};

} // namespace node

#endif // USER_NODE_HPP