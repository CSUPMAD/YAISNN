#include <cstdint>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <unordered_map>
#include <algorithm>

using namespace std;

#ifndef DEBUG
#define DEBUG 0
//#define DEBUG 1
//#define DEBUG 2
#endif

#ifndef FOUT
//#define FOUT DEBUG
#define FOUT 1
#endif

#define d_cerr  if(DEBUG>=1)cerr
#define d_cerr2 if(DEBUG>=2)cerr
#define d_cerr3 if(DEBUG>=3)cerr

typedef uint32_t nsize_t;
typedef int32_t ntime_t;

/*
struct nsize_pair_t{
nsize_t pre;
nsize_t local_post;
};
*/

constexpr int RSEED = 0;
//constexpr int RSEED = 1;
//constexpr ntime_t END_TIME = 300*1000; // msec
constexpr ntime_t END_TIME = 1000*1000; // msec


constexpr nsize_t Ne = 800;
constexpr nsize_t Ni = 200;
constexpr nsize_t N = Ne + Ni;

// Too Many Spikes Threshold (firing rate)
constexpr double TMST = 0.95;

// Too Many Spikes Threshold ( succeeded time[msec] )
constexpr ntime_t TMSTT = 5;

static ntime_t cttms = 0; // continuous time for too many spikes

constexpr double ALPHA = 0.1;

//constexpr nsize_t WANDS = N * ALPHA;
constexpr nsize_t WANDS = 100;


constexpr double SPIKE_THRESHOLD_mV = 30.0;

constexpr nsize_t TONIC_UNIT = 1000;
constexpr double TONIC_INPUT_pA = 20.0;
constexpr nsize_t TONIC_COUNT =
	((N/TONIC_UNIT) > 1) ? (N/TONIC_UNIT) : 1;


//excitatory, inhibitory (2 types)
constexpr size_t NTYPES = 2;

constexpr ntime_t MAX_DELAY[NTYPES] = {20, 1};

static double v[N];
static double u[N];
//static double I[N];

constexpr double v_init[NTYPES] = {-65, -65};
constexpr double u_init[NTYPES] = {-13, -13};

constexpr double a[NTYPES] = {0.02, 0.1};
constexpr double b[NTYPES] = {0.2, 0.2};
constexpr double c[NTYPES] = {-65, -65};
constexpr double d[NTYPES] = {8.0, 2.0};

static ntime_t tmsec = 0;

/*
template<typename T, size_t BUFSIZE>
class circular_buffer{

	T buf[BUFSIZE];

	size_t raw_index;

	inline size_t get_newest_index()
		{ return raw_index%BUFSIZE; }

	size_t get_internal_index(size_t index)
		{ return (index+raw_index)%BUFSIZE; }

public:
	void pop(){ raw_index++; }

	void push(T arg){ pop(); buf[get_newest_index()] = arg; }

	inline T& operator[](size_t index)
	{
		if(index > BUFSIZE) throw;

		return buf[get_internal_index];
	}

};
*/

template<typename Iter> constexpr Iter
	arraymax(Iter a, Iter b) noexcept {

	return (a!=b) ?
		(*(arraymax(a+1,b)) > *a ? b : a) :
		b;

}

constexpr ntime_t DELAY_TIME_WINDOW =
	*arraymax(MAX_DELAY, (MAX_DELAY+NTYPES-1));

double I[DELAY_TIME_WINDOW][N];

typedef unordered_multimap<nsize_t, pair<nsize_t, nsize_t> > nconn_t;
//typedef unordered_multimap<nsize_t, nsize_pair_t> nconn_t;

nsize_t  conn[N][WANDS];
nsize_t delay[N][WANDS];

//static nconn_t conn;
static nconn_t r_conn;


double weight[N][WANDS];
//double weight_derivative[N][WANDS];

constexpr double INIT_EXC_WEIGHT = 6.0;
constexpr double INIT_INH_WEIGHT = -5.0;
constexpr double INIT_WEIGHT_DERIVATIVE = 0.0;

void update_weight(nsize_t index);
void dump_weight(ofstream& fout);

/*
//add-STDP params
constexpr double //A_p = 0.1;
constexpr double //A_m = 0.12;

constexpr double //tau_p = 20;
constexpr double //tau_m = 20;
*/

//log-STDP Params
//constexpr double eta = 0.1;
constexpr double eta = 0.5; //modified
//constexpr double sigma = 0.6; //modified
constexpr double sigma = 0.6;
constexpr double alpha = 5;
constexpr double beta = 50;

//constexpr double j_z = 0.25;
constexpr double   j_z = 1.0; //modified

//constexpr double c_p = 1;
constexpr double   c_p = 1.0; //modified
//constexpr double c_d = 0.5;
constexpr double   c_d = 0.10; //modified

//constexpr double tau_p = 17;
constexpr double   tau_p = 45; //modified
//constexpr double tau_d = 34;
constexpr double   tau_d = 65; //modified

ntime_t mrft[N]; //most recent firing time


#pragma omp declared simd
constexpr inline ntime_t windowed_time(ntime_t t){

	return t%DELAY_TIME_WINDOW;

}

inline ntime_t get_current_time(){

	return tmsec;

}

inline void step_forward_time(){

	tmsec++;

}

#pragma omp declared simd
constexpr inline size_t itt(size_t index){
	//itt means index_to_type
	//translate index to NTYPE
	return (index<Ne) ? 0 : 1;
}

void init_vu(){

#pragma omp parallel for simd
	for (nsize_t i=0; i<N; i++){
		nsize_t j(itt(i));
		v[i] = v_init[j];
		u[i] = u_init[j];
	}

}

template<typename T1, typename T2, typename elem_t>
void insert_sort(T1 *a, T2 b, elem_t arg_beg, elem_t arg_end){

	elem_t beg(arg_beg);
	elem_t end(arg_end - 1);

	for (elem_t i = beg + 1; i <= end; i++)
		for (elem_t j = i; j > beg && a[j - 1] > a[j]; j--){
			swap(a[j], a[j - 1]);
			swap(b[j], b[j - 1]);
		}

}

void do_init_conn(
	mt19937& mt,
	nsize_t beg,
	nsize_t end,
	nsize_t post_beg,
	nsize_t post_end,
	ntime_t max_delay){

	uniform_int_distribution<nsize_t> rand(post_beg, post_end-1);
	//beg <= rand(mt) <= end-1

	for (nsize_t i=beg; i<end; i++){

		ntime_t raw_delay = 0;

		//for optimization
		//guarantee memory access locality

		nsize_t conn_unsorted[WANDS];
		ntime_t delay_unsorted[WANDS];
		nsize_t temp_conn[WANDS];
		ntime_t temp_delay[WANDS];

		//set initial value
		for(nsize_t j=0; j<WANDS; j++){

			nsize_t post_index;
			bool exists;

			do{

				exists = 0;             // avoid multiple synapses
				post_index = rand(mt);

				if (post_index==i) exists=1;
				for (nsize_t k=0; k<j; k++) if (conn_unsorted[k]==post_index) exists = 1;

			}while (exists == 1);

			ntime_t axonal_delay = raw_delay++%max_delay;

			////auto axon(make_pair(post, delay));
			////auto r_axon(make_pair(i, delay));

			////conn.insert(make_pair(i, axon));
			////r_conn.insert(make_pair(post, r_axon));

			//conn[i][j] = post;
			//delay[i][j] = axonal_delay;

			//r_conn.insert(make_pair(post, make_pair(i, j)));

			conn_unsorted[j] = post_index;
			delay_unsorted[j] = axonal_delay;

		}


		//sort arrays
		insert_sort(conn_unsorted, delay_unsorted, static_cast<nsize_t>(0), WANDS);

		//insert sorted values
		for(nsize_t j=0; j<WANDS; j++){

			conn[i][j] = conn_unsorted[j];
			delay[i][j] = delay_unsorted[j];

			r_conn.insert(make_pair(conn_unsorted[j], make_pair(i, j)));

		}


	}

}

void init_conn(){

	mt19937 mt(RSEED);

	//Ne
	{
		nsize_t beg = 0, end = Ne;
		nsize_t ntype(itt(beg));
		do_init_conn(mt, beg, end, 0, N, MAX_DELAY[ntype]);
	}

	//Ni
	{
		nsize_t beg = Ne, end = N;
		nsize_t ntype(itt(beg));
		//do_init_conn(mt, beg, end, 0, N, MAX_DELAY[ntype]);
		do_init_conn(mt, beg, end, 0, Ne, MAX_DELAY[ntype]);
	}

}

void init_weight(){

	//Ne
	for(nsize_t i=0; i<Ne; i++)
#pragma omp parallel for simd
		for(nsize_t j=0; j<WANDS; j++) {

			weight[i][j] = INIT_EXC_WEIGHT;
			//weight_derivative[i][j]
			//	= INIT_WEIGHT_DERIVATIVE;

		}

	//Ni
	for(nsize_t i=Ne; i<N; i++)
#pragma omp parallel for simd
		for(nsize_t j=0; j<WANDS; j++) {

			weight[i][j] = INIT_INH_WEIGHT;
			//weight_derivative[i][j]
			//	= INIT_WEIGHT_DERIVATIVE;

		}

}

void init_mrft(){

	const ntime_t NEGATIVE_BIG_VALUE(-100);

#pragma omp parallel for simd
	for(nsize_t i=0; i<N; i++)
		mrft[i] = NEGATIVE_BIG_VALUE;

}

#pragma omp declare simd
void clear_all_I(){

	for(ntime_t t=0; t<DELAY_TIME_WINDOW; t++)
#pragma omp parallel for simd
		for(nsize_t i=0; i<N; i++)
			I[t][i] = 0.0;

}

#pragma omp declare simd
inline void clear_I(nsize_t index){

	I[windowed_time(get_current_time())][index] = 0.0;

}

#pragma omp declare simd
inline void clear_I(){

#pragma omp parallel for simd
	for(nsize_t i=0; i<N; i++)
		clear_I(i);

}

#pragma omp declare simd
inline void input_tonic_current(
	double I[][N],
	nsize_t beg,
	nsize_t end){

		static mt19937 mt(RSEED);

		uniform_int_distribution<nsize_t>
			rand(beg, end-1);
		//beg <= rand(mt) <= end-1

		I[windowed_time(get_current_time())][rand(mt)] += TONIC_INPUT_pA;

}

void do_update_vu(
	double *v, double *u,
	double a, double b,
	const double I[][N],
	nsize_t beg, nsize_t end){

#pragma omp simd
	for(	nsize_t i = beg;
		i<end;
		i++){

		double dv, du;

		//update v
		dv = 0.5*((0.04*v[i]+5)*v[i]+140-u[i]
			+I[windowed_time(get_current_time())][i]);
		v[i] += dv;
		dv = 0.5*((0.04*v[i]+5)*v[i]+140-u[i]
			+I[windowed_time(get_current_time())][i]);
		v[i] += dv;

		//update u
		du = a * (b * v[i] - u[i]);
		u[i] += du;

	}

}

void update_vu(){

	//Ne
	{
		const nsize_t beg(0);
		const nsize_t end(Ne);

		//for calculation efficiency
		const double ae(a[itt(beg)]);
		const double be(b[itt(beg)]);

		do_update_vu(v, u, ae, be, I, beg, end);
	}

	//Ni
	{
		const nsize_t beg(Ne);
		const nsize_t end(N);

		//for calculation efficiency
		const double ai(a[itt(beg)]);
		const double bi(b[itt(beg)]);

		do_update_vu(v, u, ai, bi, I, beg, end);
	}

}

#pragma omp declare simd
constexpr inline bool is_fired(double v){

	return v > SPIKE_THRESHOLD_mV;

}

#pragma omp declare simd
void post_spike_recovery(double *v, double *u, nsize_t index){

	v[index]  = c[itt(index)];
	u[index] += d[itt(index)];

}

#pragma omp declare simd
void propagate_spike_info(nsize_t index){

///time_t axonal_delay;


#pragma omp parallel for simd
	for(nsize_t i=0; i<WANDS; i++){

		nsize_t postsynaptic_neuron_index
			= conn[index][i];

		///axonal_delay
		ntime_t axonal_delay
			= delay[index][i];

		I[windowed_time(get_current_time()+axonal_delay)]
			[postsynaptic_neuron_index]
			+= weight[index][i];

	}

///if(axonal_delay == 0) cout << weight[index][0] << endl;

}

void preupdate_vu(ofstream& fout){

#pragma omp parallel for simd
	for(nsize_t i=0; i<TONIC_COUNT; i++)
		input_tonic_current(I, 0, N);

//for debug print & logging.
//no parallelization here.
	double firing_rate = 0;

	for(nsize_t i=0; i<N; i++){

		if( is_fired(v[i]) ){

			d_cerr2 << " FIRE :" << endl;
			d_cerr2 << " TIME :" << get_current_time() << endl;
			d_cerr2 << " INDEX:" << i     << endl;
			//log to file

			if(FOUT) fout << get_current_time() << ' ' << i << endl;

			firing_rate += 1.0;

		}

	}

	firing_rate /= N;

	d_cerr3 << "TIME  : " << get_current_time() << endl;
	d_cerr3 << "FR    : " << firing_rate << endl;

	if(firing_rate > TMST) cttms += 1;
	else cttms = 0;

#pragma omp parallel for simd
	for(nsize_t i=0; i<N; i++){

		if( is_fired(v[i]) ){

			mrft[i] = get_current_time();

			update_weight(i);

			post_spike_recovery(v, u, i);

			propagate_spike_info(i);

		}

	}

}

void postupdate_vu(){

	clear_I();

}

void stepwise_calculation(ofstream& fout){

	preupdate_vu(fout);

	///cout << " pre:" << I[windowed_time(get_current_time())][11] << endl;

	update_vu();

	postupdate_vu();

	///cout << "post:" << I[windowed_time(get_current_time())][11] << endl;

}

#pragma omp declare simd
void update_weight(nsize_t index){

	static mt19937 mt(RSEED);
	static normal_distribution<> rand(0, sqrt(sigma));


	auto range = r_conn.equal_range(index);

	for( auto it=range.first; it!=range.second; it++ ){
		nsize_t pri = it->second.first; //preneuron_index
		nsize_t ci = it->second.second; //connection_index

		//update Ne only
		if(pri >= Ne) continue;

		ntime_t u = get_current_time() - (mrft[pri] + delay[pri][ci]); //tpost - tpre //diff time
		double& j = weight[pri][ci];

		if( j < 0 ) continue;

		double w(0.0);
		double noise(rand(mt));

		if( u < 0 ){
			//LTP

			double f_p = c_p * exp( -j / (j_z * beta) );
			w = f_p * exp( u / tau_p );

		}
		else if( u > 0 ){
			//LTD

			double f_m(c_d);
			if( j <= j_z ) f_m *= j / j_z;
			else f_m *= ( 1 + log( 1 + alpha * ( (j / j_z) - 1 ) ) / alpha );
			w = -f_m * exp( -u / tau_d );

			//using unreached spike information may lead to corrupt result.
			//to be considered.
		}
		//omit case u==0

		const double dj(eta * ( 1 + noise ) * w);
		const double updated_j(j+dj);
		if( j>=0 ){

			if( j + dj <=0 )
				j = 0;
			else
				j = updated_j;

		}

	}

}

void dump_weight(ofstream& fout){

	for(nsize_t i=0; i<N; i++){

		for(nsize_t j=0; j<WANDS; j++){

			if(j!=0) fout << ' ';
			fout << weight[i][j];

		}

		fout << endl;

	}

}

int main(){

	//d_cerr << *arraymax(MAX_DELAY, &(MAX_DELAY[1])) << endl;

	d_cerr << "START : init" << endl;
	ofstream out("fire.dat");
	init_vu();
	init_conn();
	init_weight();
	init_mrft();
	clear_all_I();
	d_cerr << "END   : init" << endl;

	d_cerr << "START : main calculation" << endl;

	for(; get_current_time()<END_TIME; step_forward_time()){

		stepwise_calculation(out);

		if(cttms > 0) {

			d_cerr << "TIME  : " << get_current_time() << endl;		
			d_cerr << "NOTICE: Too Many Spikes." << endl;

		}

		if(cttms > TMSTT) {

			d_cerr << "ERROR : Abort Calculation." << endl;
			break;
		}

	}

	d_cerr << "END   : main calculation" << endl;

	ofstream wout("weight.dat");
	dump_weight(wout);

	return 0;

}
