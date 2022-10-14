#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <time.h>

#include <boost/numeric/ublas/matrix.hpp>   // Matrix
#include <boost/asio.hpp>                   // TCP core
#include <boost/program_options.hpp>        // program args

#include <omp.h>

//g++ Server.cpp -o Server -lpthread -lboost_program_options -fopenmp

using namespace std;
namespace basio  = boost::asio;
namespace bip    = boost::asio::ip;
namespace buplas = boost::numeric::ublas;
namespace po     = boost::program_options;

// gen to file
bool WRITE_FILE = true;

void gen_matrices(string &full_file, double megabytes_size = 1) {

    const char *tmp_file = "./.gen_tmp";
    if (WRITE_FILE)
        cout << "Generate matrices to " << tmp_file << endl;

    // transform Mb to Bytes
    size_t bytes_size = megabytes_size * 1024 * 1024;

    // gen size
    srand(time(NULL));
    int size = rand() % 5 + 2;
    string size_str = to_string(size);

    // remove and create file
    std::remove(tmp_file);
    ofstream tmp(tmp_file);

    // write size
    full_file+=(size_str + '\n');
    if (WRITE_FILE)
        tmp << (size_str + '\n');

    double lower_bound = 0;
    double upper_bound = 10000;
    std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
    std::default_random_engine re;

    // gen matrices
    while (full_file.size() <= bytes_size) {
        #pragma omp parallel for num_threads(size)
        for (int i = 1; i <= size; i++) {
            string buf;
            for (int j = 1; j <= size; j++) {
                double random_double = unif(re);
                buf+=to_string(random_double);

                // add delimeter
                if (j == size)
                    buf+='\n';
                else
                    buf+=' ';
            }

            #pragma omp critical
            {
                full_file+=buf;
                if (WRITE_FILE)
                    tmp << buf;
            }
        }
    }

    tmp.close();
}


int read_matrix_from_file(string file_name, string &full_file) {
    cout << "Read matrices from " << file_name << endl;
    ifstream input_file(file_name);

    if (input_file.is_open()) {
        char buf;
        while (! input_file.eof()) {
            input_file.get(buf);
            full_file+=buf;
        }
    } else {
        cerr << "Wrong file: " << file_name << endl;
        exit(1);
    }

    // remove artifact
    full_file.erase(full_file.size() - 1, 1);

    input_file.close();
    return 0;
}


// make string from input tcp buffer
string make_string(basio::streambuf& streambuf) {
    return {buffers_begin(streambuf.data()), buffers_end(streambuf.data())};
}

int main(int argc, char* argv[]) {
    // GLOBAL VARS
    int PORT = 55555;
    string FILE = "./Matrices.txt";
    double SIZE = 1;
    string full_file;

    // ARGS
    // ------------------------------------------------------------------------
    // Declare the supported options.

    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "print help")
        ("port,p", po::value<int>(), "set port [default: 55555]")
        ("file,f", po::value<string>(), "set matrices file")
        ("size,s", po::value<double>(), "set stream size in Mb [default: 1]")
        ("write,w", po::value<bool>(), "write gen in file (0/1) [default: 1]");


    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    if (vm.count("port"))
        PORT = vm["port"].as<int>();

    if (vm.count("size"))
        SIZE = vm["size"].as<double>();

    if (vm.count("write"))
        WRITE_FILE = vm["write"].as<bool>();

    if (vm.count("file")) {
    // File specified -> read from file
        FILE = vm["file"].as<string>();
        read_matrix_from_file (FILE, full_file);
    } else {
    // File not specified -> gen matrices string
        gen_matrices (full_file, SIZE);
    }
    // ------------------------------------------------------------------------

    // SET UP TCP
    // ------------------------------------------------------------------------
    basio::io_context io_context;
    boost::system::error_code error;
    // accept input connection
    bip::tcp::acceptor acceptor(io_context,
                                bip::tcp::endpoint(bip::tcp::v4(), PORT));

    cout << "[Server] Wait for client ..." << endl;
    bip::tcp::socket socket(io_context);
    acceptor.accept(socket);

    // Size of TCP message
    int HEADER_SIZE = 8192;
    vector<char> buf(HEADER_SIZE);
    int ACK_SIZE = 3;
    // ------------------------------------------------------------------------

    // TRANSFER FILE AS STRING
    // ------------------------------------------------------------------------
    cout << "[Server] Send message size ..." << endl;
    string msg_size_str = to_string(full_file.size());
    basio::write(socket, basio::buffer(msg_size_str, HEADER_SIZE), error);

    cout << "[Server] Receive ack ..." <<  endl;
    basio::read(socket, basio::buffer(buf, ACK_SIZE), error);
    if (error.value() != boost::system::errc::success)
        throw boost::system::system_error(error);

    cout << "[Server] Send message ..." << endl;
    basio::write(socket, basio::buffer(full_file, full_file.size()), error);
    // ------------------------------------------------------------------------

    // RECEIVE
    // ------------------------------------------------------------------------
    buf.resize(HEADER_SIZE);
    cout << "[Server] Recieve message size ..." <<  endl;
    //basio::read(socket, basio::buffer(buf, HEADER_SIZE), error);
    socket.read_some(basio::buffer(buf, HEADER_SIZE), error);
    if (error.value() != boost::system::errc::success)
        throw boost::system::system_error(error);
    int msg_size = stoi(buf.data());
    cout << "[Server] Message size: " << msg_size << endl;

    cout << "[Server] Send ack ..." << endl;
    basio::write(socket, basio::buffer("ACK", ACK_SIZE), error);

    buf.resize(msg_size);
    cout << "[Server] Recieve message ..." <<  endl;
    basio::read(socket, basio::buffer(buf, msg_size), error);
    if (error.value() != boost::system::errc::success)
        throw boost::system::system_error(error);
    string input(buf.begin(), buf.end());
    // ------------------------------------------------------------------------

    // write answer to file
    string answer_file_name = socket.remote_endpoint().address().to_string()
                                + "_answer.txt";
    ofstream answer_file(answer_file_name);
    answer_file << input;
    answer_file.close();
    cout << "See answer in " << answer_file_name << endl;

    cout << "[Server] Finish" <<  endl;
    return 0;
}
