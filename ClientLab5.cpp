/*
 * File:   Main.cpp
 * Author: maslov_a
 *
 * Created on 17 сентября 2022 г., 13:56
 */

//# mpic++ ClientLab3.cpp -o ClientLab3 -lpthread -lboost_program_options
// mpic++ -o ClientLab5.o -c ClientLab5.cpp
// mpic++ -o ClientLab5 ClientLab5.o ClientLab5CUDA.o -L/usr/local/cuda/lib64 -lpthread -lboost_program_options -lcuda -lcudart
// mpirun -np 4 ./ClientLab5

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

#include <boost/asio.hpp>                   // TCP
#include <boost/numeric/ublas/matrix.hpp>   // Matrix
#include <boost/array.hpp>                  // Matrix
#include <boost/program_options.hpp>        // program args

// Open MPI
#include <mpi.h>


using namespace std;
namespace basio  = boost::asio;
namespace bip    = boost::asio::ip;
namespace buplas = boost::numeric::ublas;
namespace po     = boost::program_options;

bool PROGRESS_BAR = false;

vector<vector<double>> get_max_cuda(vector<buplas::matrix<double>> &matrix_vector,
                  chrono::high_resolution_clock::duration &duration);

buplas::matrix<double> make_matrix_from_2d_array(double **array,
        int height, int weight) {

    buplas::matrix<double> answer(height, weight);
    for (int j1 = 0; j1 < height; j1++)
        for (int j2 = 0; j2 < weight; j2++)
            answer(j1, j2) = array[j1][j2];

    return answer;
}

double **make_2d_array_from_matrix(buplas::matrix<double> &matr,
        double **answer, int height, int weight) {

    for (int j1 = 0; j1 < height; j1++)
        for (int j2 = 0; j2 < weight; j2++)
            answer[j1][j2] = matr(j1, j2);

    return answer;
}

double **make_2d_array_from_2d_vector(vector<vector<double>> &result_vector,
        double **answer, int height, int weight) {

    for (int j1 = 0; j1 < height; j1++) {
        for (int j2 = 0; j2 < weight; j2++) {
            answer[j1][j2] = result_vector[j1][j2];
        }
    }

    return answer;
}

vector<vector<double>> make_2d_vector_from_2d_array(double **array,
        int height, int weight) {

    vector<double> row(weight);
    vector<vector<double>> answer(height);

    for (int j1 = 0; j1 < height; j1++) {
        for (int j2 = 0; j2 < weight; j2++)
            row[j2] = array[j1][j2];
        answer[j1] = row;
    }

    return answer;
}

// create contiguous 2d array
double **alloc_2d_double(int h, int w) {
    double *row = (double *)malloc(h * w * sizeof(double));
    double **array = (double **)malloc(h * sizeof(double*));
    for (int i = 0; i < h; i++)
        array[i] = &(row[w * i]);

    return array;
}

void unalloc_2d_double(double** array) {
    free(array[0]);
    free(array);
}

string gen_answer(vector<vector<double>> result_vector,
        chrono::high_resolution_clock::duration &duration) {
    string answer_str;

    // start measure
    //TODO: use monolitic clock
    chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();

    cout << "[Client] Generate answer ..." << endl;
    for (int i = 0; i < result_vector.size(); i++) {
        // header of matrix
        answer_str+=("Matrix[" + to_string(i) + "]\n");
        for (int j = 0; j < result_vector[i].size(); j++) {
            answer_str+=(to_string(result_vector[i][j]) + '\n');
        }
    }

    // stop measure
    chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    return answer_str;
}

vector<vector<double>> get_max(vector<buplas::matrix<double> > &matrix_vector,
        chrono::high_resolution_clock::duration &duration){

    //cout << "[Client] Calc max numbers ..." << endl;

    // start measure
    chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();

    int vector_size = matrix_vector.size();
    vector<vector<double>> answer(vector_size);
    int matrix_size = matrix_vector[0].size1();

    for (int i = 0; i < vector_size; i++) {
        vector<double> max_lines(matrix_size);
        for (int h = 0; h < matrix_size; h++) {
            vector<double> line(matrix_size);
            for (int l = 0; l < matrix_size; l++) {
                line[l] = matrix_vector[i](h, l);
            }

            double max = *max_element(line.begin(), line.end());
            double result = max * max;
            max_lines[h] = result;
        }

        answer[i] = max_lines;
        // progress bar
        if (PROGRESS_BAR)
            printf("%d matrices of %d\r", i, vector_size);
    }

    // stop measure
    chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    //cout << endl;

    return answer;
}

// make vector of matrix from string
vector<buplas::matrix<double> >parse_input (string input,
        chrono::high_resolution_clock::duration &duration) {
    vector< buplas::matrix<double> > matrix_vector;

    // find first line (size of matrix)
    int size_pos = input.find('\n');
    int size     = stoi(input.substr(0, size_pos));
    int pos      = size_pos + 1;
    cout << "[Client] Matrix size = " << size <<  endl;

    int h, l;                                   // current coords of matrix
    int num_pos;                                // number position
    string num_str;                             // found number string
    double num;                                 // found number

    // start measure
    chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
    int input_size = input.size();

    cout << "[Client] Parse input ... "  <<  endl;
    while (pos < input_size) {
        // create matrix
        buplas::matrix<double> matrix(size,size);

        // parse height
        h = 0;
        while ( h < size ) {
            // parse length
            l = 0;
            while ( l < size ) {
                num_pos = input.find_first_of(" \n\0", pos);
                if (num_pos <= 0)
                    num_pos = input.size();
                num_str = input.substr(pos, num_pos - pos);
                num = stod(num_str);
                pos = num_pos + 1;

                matrix.insert_element(h, l , num);
                l++;
            }
            h++;
        }

        // add matrix
        matrix_vector.push_back(matrix);

        // progress bar
        if (PROGRESS_BAR)
            printf("%d bytes of %d\r", pos, input_size);
    }

    // stop measure
    chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    cout << endl;

    return matrix_vector;
}

// process args
int get_args(po::options_description &desc, int &P, string &H,
                                            int &ac, char **av) {
    desc.add_options()
        ("help,h", "print help")
        ("port,p", po::value<int>(),    "set port [default: 55555]")
        ("host,H", po::value<string>(), "set host ip [default: 127.0.0.1]")
        ("progress,P", po::value<bool>(), "progress bar 1/0 [default: 1]");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        exit(1);
    }

    if (vm.count("port"))
        P = vm["port"].as<int>();

    if (vm.count("host"))
        H = vm["host"].as<string>();

    if (vm.count("progress"))
        PROGRESS_BAR = vm["progress"].as<bool>();

    return 0;
}


int main(int argc, char* argv[]) {

    // mpi vars
    // --------
    // rank and size
    int RANK_MPI, SIZE_MPI;
    // define MASTER
    int MASTER = 0;
    // MPI tags
    int SIZE_VECTOR_TAG = 0;
    int MATRIX_TAG      = 1;
    int SIZE_MATRIX_TAG = 2;
    int RESULT_SIZE_TAG = 3;
    int RESULT_TAG      = 4;
    int BLOCK_TAG       = 5;
    // parts
    int part, remainder;
    // --------

    // start MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &SIZE_MPI);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK_MPI);

    if (SIZE_MPI == 1) {
        cerr << "1 thread???" << endl;
        return 1;
    }

    // times
    chrono::high_resolution_clock::duration parse_time, gen_answer_time, calcs_time;
    long double parse_time_s, gen_answer_time_s, calcs_time_s;

    int msg_size;       // size of tcp msg
    int full_part_size; // size of matrices vector part
    int matrix_size;
    int vector_size;    // size of matrices vector part
    int row_size;       // size of row answer vector

    // Size of TCP message
    int HEADER_SIZE = 8192;
    int ACK_SIZE = 3;

    // GLOBAL VARS
    int PORT    = 55555;
    string HOST = "127.0.0.1";

    // process args
    po::options_description desc("Options");
    get_args(desc, PORT, HOST, argc, argv);

    // Connect backend
    basio::io_context io_context;           // core IO object
    bip::tcp::endpoint ep(bip::address::from_string(HOST), PORT);
    bip::tcp::socket socket(io_context);
    boost::system::error_code error;        // contain asio error

    // sync on master process
    if (RANK_MPI == MASTER) {
        vector<char> buf(HEADER_SIZE);

        // try to connect to socket
        try {
            socket.connect(ep);
        } catch (const boost::system::system_error& ex) {
            cerr << "[Client error] Start server at first!" << endl;
            exit(1);
        }

    // RECEIVE
    // -------------------------------------------------------------------------
        socket.read_some(basio::buffer(buf, HEADER_SIZE), error);
        if (error.value() != boost::system::errc::success)
            throw boost::system::system_error(error);
        msg_size = stoi(buf.data());
        cout << "[Client] Message size: " << msg_size << " bytes "<< endl;

        // Send ack
        basio::write(socket, basio::buffer("ACK", ACK_SIZE), error);

        buf.resize(msg_size);
        cout << "[Client] Recieve message ..." <<  endl;
        basio::read(socket, basio::buffer(buf, msg_size), error);
        if (error.value() != boost::system::errc::success)
            throw boost::system::system_error(error);
        string input(buf.begin(), buf.end());
    // -------------------------------------------------------------------------

    // CALCS
    // -------------------------------------------------------------------------
        // parse matrices
        cout << "[Client] Stream size: " << input.size() << " bytes" << endl;
        vector<buplas::matrix<double> > matrix_vector = parse_input(input, parse_time);
        cout << "[Client] Matrices number = " << matrix_vector.size() << endl;
        // time
        parse_time_s = parse_time.count()*1e-9;
        cout << "Parse time: " << parse_time_s << " s" << endl;

        // --- send partitions ---
        // divide data for MPI (one half for CUDA)
        //part = matrix_vector.size() / ((SIZE_MPI - 1) * 2);
        //part = matrix_vector.size() / SIZE_MPI;
        part = matrix_vector.size() / (SIZE_MPI + 1);
        cout << "[Open MPI] part = " << part << endl;
        // assign remainder to the last rank (thread)
        //remainder = matrix_vector.size() % ((SIZE_MPI - 1) * 2);
        //remainder = matrix_vector.size() % SIZE_MPI;
        remainder = matrix_vector.size() % (SIZE_MPI + 1);
        cout << "[Open MPI] remainder = " << remainder << endl;

        // beauty output
        printf("%-28s", "[Client 0]: Send part to ");
        for (int slave = 0; slave < SIZE_MPI; slave++ )
            printf("%-8d ", slave);
        printf("\n%28s", " ");
        // create partitions and send them
        for (int slave = 0; slave < SIZE_MPI; slave++ ) {
            vector<buplas::matrix<double>>::const_iterator start = matrix_vector.begin() + slave * part;
            vector<buplas::matrix<double>>::const_iterator fin;

            // correct filling (remainder for CUDA)
            if (slave == SIZE_MPI - 1) {
                //fin = matrix_vector.begin() + slave * part + part * slave + remainder;
                //fin = matrix_vector.begin() + slave * part + part + remainder;
                fin = matrix_vector.begin() + slave * part + part * 2 + remainder;
            }
            else
                fin = matrix_vector.begin() + slave * part + part;

            vector<buplas::matrix<double>> matrix_vector_part(start, fin);

            // size of send data to each slave
            matrix_size = matrix_vector_part[0].size1();
            vector_size = matrix_vector_part.size();

            // send matrix size
            MPI_Send(&matrix_size, 1, MPI_INT, slave, SIZE_MATRIX_TAG, MPI_COMM_WORLD);

            // send vector size
            printf("%-9d", vector_size);
            MPI_Send(&vector_size, 1, MPI_INT, slave, SIZE_VECTOR_TAG, MPI_COMM_WORLD);

            // send vector of matrices
            double **snd_2d_array = alloc_2d_double(matrix_size, matrix_size);
            //
            for (int i = 0; i < vector_size; i++) {
                snd_2d_array = make_2d_array_from_matrix(matrix_vector_part[i],
                        snd_2d_array, matrix_size, matrix_size);
                // send as 2d array
                MPI_Send(&(snd_2d_array[0][0]),
                        matrix_size * matrix_size,
                        MPI_DOUBLE,
                        slave,
                        MATRIX_TAG,
                        MPI_COMM_WORLD);
            }
            //
            unalloc_2d_double(snd_2d_array);
        }
        printf("\n");
    } else
        socket.close(); // dont use socket on slaves

    // wait for master process
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Status ignored_status;

    // recv matrix size
    MPI_Recv(&matrix_size, 1, MPI_INT, MASTER, SIZE_MATRIX_TAG,
            MPI_COMM_WORLD, &ignored_status);

    // recv vector size
    MPI_Recv(&vector_size, 1, MPI_INT, MASTER, SIZE_VECTOR_TAG,
            MPI_COMM_WORLD, &ignored_status);

    // receive vector of matrices
    vector<buplas::matrix<double>> matrix_vector_part(vector_size); // vector of matrices
    // init recv 2d array
    double **recv_2d_array = alloc_2d_double(matrix_size, matrix_size);
    //
    for (int i = 0; i < vector_size; i ++) {
        MPI_Recv(&(recv_2d_array[0][0]), matrix_size * matrix_size, MPI_DOUBLE, MASTER,
                MATRIX_TAG, MPI_COMM_WORLD, &ignored_status);

        matrix_vector_part[i] = make_matrix_from_2d_array(recv_2d_array, matrix_size, matrix_size);
    }
    //
    unalloc_2d_double(recv_2d_array);

    // main buisness calcs (last RANK = CUDA thread)
    vector<vector<double>> result_vector;
    if (RANK_MPI == SIZE_MPI - 1)
        result_vector = get_max_cuda(matrix_vector_part, calcs_time);
    else
        result_vector = get_max(matrix_vector_part, calcs_time);
    //
    calcs_time_s = calcs_time.count()*1e-9;
    cout << "[Client " << RANK_MPI << "] Calcs time: " << calcs_time_s << " s" << endl;

    // size info vector of vectors
    vector_size     = result_vector.size();
    row_size        = result_vector[0].size();
    full_part_size  = vector_size * row_size;

    MPI_Barrier(MPI_COMM_WORLD);

    if (RANK_MPI == MASTER) {
        // collect all data from slaves
        for (int i = 1; i < SIZE_MPI; i++) {
            // recv size
            MPI_Recv(&full_part_size, 1, MPI_INT, i, RESULT_SIZE_TAG, MPI_COMM_WORLD, &ignored_status);

            // buf 2d array here, keep remainder in mind!!!
            double **snd_ans_2d_array = alloc_2d_double(full_part_size / row_size, row_size);
            vector<vector<double>> tmp_result_vector;

            // recv 2d vector
            MPI_Recv(&(snd_ans_2d_array[0][0]),
                    full_part_size,
                    MPI_DOUBLE,
                    i,
                    RESULT_TAG,
                    MPI_COMM_WORLD,
                    &ignored_status);
            tmp_result_vector = make_2d_vector_from_2d_array(snd_ans_2d_array,
                    full_part_size / row_size, row_size);

            unalloc_2d_double(snd_ans_2d_array);
            result_vector.insert(result_vector.end(), tmp_result_vector.begin(), tmp_result_vector.end());
        }

        vector<char> buf(HEADER_SIZE);

        // gen string from answer
        string ANSWER = gen_answer(result_vector, gen_answer_time);
        gen_answer_time_s = gen_answer_time.count()*1e-9;
        cout << "Gen answer time: " << gen_answer_time_s << " s"  << endl;

        // add times to answer
        string parse_time_str      = "Parse time, s:      ";
        string calcs_time_str      = "Calcs time, s:      ";
        string gen_answer_time_str = "Gen answer time, s: ";
        parse_time_str      += (to_string(parse_time_s) + "\n");
        calcs_time_str      += (to_string(calcs_time_s) + "\n");
        gen_answer_time_str += (to_string(gen_answer_time_s) + "\n");
        ANSWER += (parse_time_str + calcs_time_str + gen_answer_time_str);
    // -------------------------------------------------------------------------

    // SEND
    // -------------------------------------------------------------------------
        msg_size = ANSWER.size();
        //cout << "[Client] Send message size ..." << endl;
        string msg_size_str = to_string(msg_size);
        basio::write(socket, basio::buffer(msg_size_str, HEADER_SIZE), error);

        buf.resize(HEADER_SIZE);
        //cout << "[Client] Receive ack ..." <<  endl;
        basio::read(socket, basio::buffer(buf, ACK_SIZE), error);
        if (error.value() != boost::system::errc::success)
            throw boost::system::system_error(error);

        cout << "[Client] Send message ..." << endl;
        basio::write(socket, basio::buffer(ANSWER, msg_size), error);
    // -------------------------------------------------------------------------

        cout << "[Client] Finish" << endl;
    }

    // send result to master process
    if (RANK_MPI != MASTER) {
        int block = 1; // serial execution

        // wait unblock from previous thread (MEMORY OPTIMIZATION)
        if (RANK_MPI != MASTER + 1)
            MPI_Recv(&block, 1, MPI_INT, RANK_MPI - 1, BLOCK_TAG, MPI_COMM_WORLD, &ignored_status);

        // send size
        MPI_Send(&full_part_size, 1, MPI_INT, MASTER, RESULT_SIZE_TAG, MPI_COMM_WORLD);

        // buf 2d array
        double **snd_ans_2d_array = alloc_2d_double(vector_size, row_size);

        // create temp 2d array
        snd_ans_2d_array = make_2d_array_from_2d_vector(result_vector,
                snd_ans_2d_array, vector_size, row_size);

        // send vector
        MPI_Send(&(snd_ans_2d_array[0][0]),
                full_part_size,
                MPI_DOUBLE,
                MASTER,
                RESULT_TAG,
                MPI_COMM_WORLD);

        unalloc_2d_double(snd_ans_2d_array);

        // unblock next thread (except last thread) (MEMORY OPTIMIZATION)
        if (RANK_MPI != SIZE_MPI - 1)
            MPI_Send(&block, 1, MPI_INT, RANK_MPI + 1, BLOCK_TAG, MPI_COMM_WORLD);
    }

    // stop MPI
    MPI_Finalize();

    return 0;
}

