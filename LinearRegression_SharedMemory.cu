// Implementing linear regression from scratch in CUDA


#include <stdio.h>
#include <cuda_runtime.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Kernel to calculate coefficients
// Calculates numerator and denominator which are then used to calculate slope and bias
__global__ void calculatePartialCoefficients(const int* x, const int* y, const int x_mean, const int y_mean, float* num, float* dem, const int n) {
    extern __shared__ float cc_shared_mem[];
    float* num_shared = cc_shared_mem;
    float* dem_shared = cc_shared_mem + blockDim.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Initialize shared memory
    num_shared[tid] = 0.0f;
    dem_shared[tid] = 0.0f;

    // Calculate partial results
    if (idx < n) {
        float x_diff = x[idx] - x_mean;
        float y_diff = y[idx] - y_mean;

        num_shared[tid] = x_diff * y_diff;
        dem_shared[tid] = x_diff * x_diff;
    }
    __syncthreads();

    // Block-wise reduction to sum partial results
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            num_shared[tid] += num_shared[tid + stride];
            dem_shared[tid] += dem_shared[tid + stride];
        }
        __syncthreads();
    }

    // Atomic operations to accumulate the block's result to global memory
    if (tid == 0) {
        atomicAdd(num, num_shared[0]);
        atomicAdd(dem, dem_shared[0]);
    }
}

// Kernel to calculate partial sums of x and y
// Calculates sum which is then used to calculate mean
__global__ void calculatePartialSums(const int* x, const int* y, int* x_partial_sum, int* y_partial_sum, const int n) {
    extern __shared__ int shared_mem[];
    int* x_shared = shared_mem;
    int* y_shared = shared_mem + blockDim.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Initialize shared memory
    x_shared[tid] = (idx < n) ? x[idx] : 0;
    y_shared[tid] = (idx < n) ? y[idx] : 0;
    __syncthreads();

    // Perform block-wise reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            x_shared[tid] += x_shared[tid + stride];
            y_shared[tid] += y_shared[tid + stride];
        }
        __syncthreads();
    }

    // Write block's partial sum to global memory
    if (tid == 0) {
        atomicAdd(x_partial_sum, x_shared[0]);
        atomicAdd(y_partial_sum, y_shared[0]);
    }
}

// Kernel to calculate the Mean Square Error (MSE)
// Calculates squared error which is then used to calculate mean squared error
__global__ void calculatePartialMSE(const int* y, const int* predictions, float* mse, int n) {
    extern __shared__ float mse_shared_mem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    mse_shared_mem[tid] = 0.0f;
    __syncthreads();

    // Calculate squared difference and store in shared memory
    if (i < n) {
        int diff = y[i] - predictions[i];
        mse_shared_mem[tid] = diff * diff;
    }
    __syncthreads();

    // Perform reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            mse_shared_mem[tid] += mse_shared_mem[tid + stride];
        }
        __syncthreads();
    }

    // Atomic operations to accumulate the block's result to global memory
    if (tid == 0) {
        atomicAdd(mse, mse_shared_mem[0]);
    }
}


// Uses mx+b to make predictions for dataset x
__global__ void makePredictions(const int* x, int* predictions, const int slope, const int bias, const int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        predictions[idx] = slope * x[idx] + bias;
    }
}

int main() {
    printf ("Linear regression from scratch in CUDA. Dataset in .cu file\n\n");

    // The Dataset
    const int N = 1024;
    int h_x[N] = { 2260, 2981, 56, 747, 2724, 1475, 198, 1313, 2185, 1461, 1526, 4026, 1672, 2155, 3615, 445, 4039, 1065, 1700, 3218, 788, 2243, 2226, 1583, 2978, 1495, 2489, 2680, 3189, 155, 172, 1187, 834, 2727, 510, 776, 777, 3747, 853, 2320, 2544, 2132, 3675, 3517, 3259, 2326, 3269, 319, 464, 3897, 2488, 63, 3133, 3217, 522, 2675, 1261, 2705, 1798, 3370, 3942, 1485, 2130, 3413, 1546, 1921, 3974, 4035, 3921, 2322, 3282, 2475, 3825, 2661, 251, 4064, 1115, 1866, 2426, 1082, 508, 2200, 2723, 1144, 1824, 3161, 2725, 3207, 1468, 625, 2804, 1915, 3506, 1049, 2727, 4077, 2107, 2666, 1042, 2457, 1610, 2817, 1813, 394, 150, 2072, 110, 3355, 3184, 1295, 2063, 371, 2961, 1539, 93, 1250, 3004, 2852, 1425, 2759, 576, 2040, 846, 3464, 1191, 1864, 191, 3788, 984, 2745, 1597, 2564, 4037, 3850, 2398, 2130, 961, 2797, 1942, 1872, 721, 1601, 2740, 969, 2335, 20, 3695, 1299, 1077, 2252, 1057, 251, 1710, 3635, 2976, 400, 3972, 3357, 304, 1261, 2399, 746, 336, 1952, 57, 3649, 1468, 939, 128, 3151, 2776, 3168, 1585, 191, 3108, 1539, 893, 3759, 1138, 1864, 2205, 738, 1340, 3285, 1715, 3684, 172, 3047, 1372, 301, 962, 277, 816, 450, 3918, 678, 3007, 3312, 2573, 4093, 1255, 732, 2755, 3805, 1585, 2728, 3057, 728, 1114, 3271, 2625, 3373, 2113, 1292, 3525, 3424, 3696, 2651, 3438, 3218, 2769, 3747, 2585, 3645, 3707, 3726, 1319, 2896, 708, 2347, 2524, 1423, 2454, 2573, 715, 3064, 3691, 3818, 2957, 3266, 719, 545, 3099, 2675, 3239, 848, 1498, 3599, 347, 3407, 3481, 3925, 3344, 3995, 1012, 132, 3195, 3625, 151, 3632, 3631, 1545, 2541, 2162, 3712, 3771, 1674, 1096, 3251, 2113, 2804, 2318, 2845, 2132, 1756, 1508, 3325, 3618, 2796, 1261, 930, 32, 1533, 3583, 2729, 1546, 3346, 2021, 1315, 155, 608, 2257, 1509, 373, 1959, 1297, 2873, 54, 1799, 2208, 1328, 2695, 760, 2040, 2215, 394, 873, 4018, 1822, 4065, 123, 3905, 230, 1384, 2876, 263, 1687, 4015, 2387, 692, 1009, 3357, 1913, 3352, 724, 3959, 67, 2128, 630, 1334, 521, 3487, 2305, 1513, 1265, 2738, 1630, 3742, 160, 2830, 216, 1059, 2270, 155, 759, 1180, 1399, 3250, 1623, 261, 298, 3012, 1943, 3113, 655, 1125, 831, 2111, 3239, 2885, 3582, 977, 1015, 2059, 2495, 800, 3719, 3592, 3804, 3861, 2676, 284, 2725, 3843, 3162, 346, 109, 3058, 1151, 3389, 3238, 3670, 3696, 3518, 3147, 2396, 2261, 3299, 2666, 1114, 325, 1916, 146, 2433, 727, 4012, 51, 2239, 815, 2134, 3459, 3866, 70, 3189, 32, 3173, 2289, 2843, 1901, 327, 3118, 2994, 2905, 3266, 3983, 1870, 1357, 2948, 585, 2076, 308, 1545, 1607, 1406, 4060, 2865, 2060, 1653, 2062, 3453, 2726, 668, 401, 2270, 2530, 2, 2050, 3015, 479, 90, 1294, 3514, 2920, 3675, 3460, 1429, 3729, 300, 1087, 2133, 1037, 430, 1924, 3551, 2707, 754, 2348, 1699, 3876, 2446, 1368, 1789, 1093, 2149, 599, 20, 960, 104, 2312, 2628, 3925, 807, 2533, 35, 2850, 1672, 1737, 2017, 1938, 3080, 1337, 2121, 2461, 2294, 207, 3931, 2903, 84, 37, 967, 3040, 1269, 1880, 1963, 1251, 1544, 1664, 673, 629, 3987, 598, 3820, 125, 3333, 843, 2642, 164, 97, 2061, 1072, 1737, 3273, 341, 2050, 2448, 1117, 2986, 173, 1114, 179, 2238, 2999, 259, 817, 4072, 1095, 2217, 321, 1350, 578, 2584, 916, 1136, 1975, 3750, 3988, 3107, 679, 1250, 205, 1965, 1691, 507, 616, 1801, 2690, 3276, 1043, 601, 2683, 3687, 1369, 2401, 1560, 947, 1775, 820, 3758, 3760, 1523, 3916, 3668, 3481, 3349, 2601, 355, 3693, 1552, 2326, 2365, 3914, 611, 1992, 390, 3946, 3284, 1125, 1634, 2050, 4000, 314, 3342, 3784, 3000, 2651, 2921, 2783, 988, 757, 601, 1879, 272, 3239, 1247, 223, 553, 1222, 2341, 1405, 2041, 1044, 2229, 3372, 1849, 450, 2539, 2825, 2124, 1961, 3372, 2152, 145, 2794, 1096, 431, 1876, 1047, 3418, 2645, 2899, 2433, 4052, 2505, 313, 3726, 3137, 3134, 696, 1825, 2397, 159, 209, 4040, 2233, 522, 860, 1781, 1697, 2897, 487, 2618, 3386, 1568, 1961, 3881, 2281, 1822, 3599, 354, 1593, 1611, 1297, 278, 4029, 3985, 908, 457, 3098, 2621, 2745, 1375, 2207, 1443, 1183, 1004, 749, 2561, 2434, 1166, 2188, 3154, 3950, 266, 425, 3906, 3011, 2970, 896, 2633, 150, 3765, 3838, 1405, 3881, 1199, 3316, 3247, 180, 286, 3000, 1854, 173, 1283, 3899, 3364, 159, 2699, 1310, 218, 1384, 281, 3151, 2936, 2123, 1127, 2182, 1324, 3643, 94, 3176, 609, 1027, 1987, 1863, 1836, 2551, 4060, 1032, 1858, 179, 4072, 1764, 1752, 1862, 809, 2744, 2521, 3930, 2658, 1650, 1351, 1611, 3670, 3214, 319, 1005, 1476, 730, 3669, 3665, 1842, 4083, 1431, 469, 1463, 1196, 906, 3515, 850, 2439, 552, 3641, 3947, 3371, 2068, 192, 3894, 277, 3223, 139, 1950, 3373, 3738, 3489, 2864, 3536, 1858, 523, 2423, 700, 1607, 524, 473, 3743, 3263, 48, 3432, 2912, 2218, 3910, 1731, 977, 1044, 3522, 2616, 867, 3222, 3199, 3315, 3321, 3356, 1455, 1944, 3986, 710, 1689, 956, 1665, 3199, 2932, 39, 2520, 426, 681, 2484, 1342, 1947, 896, 1773, 2018, 2461, 3509, 3307, 2121, 3992, 2453, 3605, 639, 3192, 3628, 2741, 2715, 913, 696, 67, 2998, 441, 1077, 711, 3687, 3063, 344, 281, 3232, 3825, 2365, 1780, 3024, 2711, 3365, 2492, 3719, 1536, 2164, 1084, 1217, 2027, 3061, 2029, 3516, 2362, 881, 1304, 2819, 1148, 2548, 1007, 1057, 1634, 2343, 1456, 3217, 2341, 3988, 3599, 2670, 1759, 2708, 2420, 3744, 3322, 114, 2429, 2766, 1973, 31, 128, 1772, 1132, 282, 472, 1697, 2558, 3435, 1468, 1631, 1084, 2596, 1769, 542, 221, 2038, 3135, 1020, 49, 3724, 2414, 3815, 2033, 2329, 1425, 3155, 2548, 642, 3842, 676, 2498, 2840, 3985, 2875, 2534, 839, 1532, 1742, 1182, 614, 2675, 2301, 497, 1567, 2001, 67, 1767, 1820, 353, 3958, 3934, 3266, 987, 2442, 3407, 3735, 79, 3104, 775, 1868, 654, 3265, 988, 1942, 2848, 904, 2251, 4047, 2223, 2104, 96, 1202, 2488, 3355, 1177, 3525, 591, 2432, 4095, 2581, 670, 3728, 1551, 73, 2703, 644, 980, 3409, 3828, 2768, 1357, 1427, 1399, 3670, 3073, 579, 2121, 2089, 3927, 2239, 3828, 1214, 1738, 1047, 17, 357, 825, 1340, 90, 622, 3786, 232, 1035, 204, 4079, 450, 3742, 845, 2625, 3615, 1942, 1186, 3584, 1022, 1180, 340, 1873, 2356, 3987, 2625, 875, 3103, 2646, 1325, 1272, 1328, 2672, 162, 2357, 3067, 2036, 3722, 3299, 1540, 1855, 3386, 2346, 2885 };
    int h_y[N] = { 56552, 74575, 1444, 18719, 68145, 36923, 5003, 32876, 54678, 36577, 38200, 100697, 41849, 53921, 90413, 11177, 101025, 26675, 42542, 80496, 19746, 56118, 55698, 39626, 74502, 37424, 62277, 67045, 79781, 3917, 4348, 29719, 20899, 68222, 12794, 19451, 19476, 93726, 21375, 58048, 63646, 53350, 91917, 87974, 81527, 58195, 81778, 8023, 11647, 97474, 62247, 1618, 78374, 80474, 13102, 66917, 31580, 67673, 45003, 84299, 98597, 37168, 53293, 85378, 38704, 48077, 99399, 100922, 98072, 58093, 82094, 61924, 95681, 66582, 6327, 101657, 27929, 46701, 60707, 27099, 12754, 55057, 68123, 28643, 45650, 79063, 68181, 80220, 36746, 15678, 70150, 47922, 87709, 26273, 68223, 101978, 52731, 66701, 26099, 61476, 40299, 70481, 45377, 9899, 3793, 51854, 2798, 83930, 79651, 32421, 51623, 9315, 74079, 38509, 2379, 31293, 75147, 71346, 35674, 69031, 14454, 51053, 21201, 86660, 29821, 46653, 4826, 94757, 24657, 68660, 39972, 64156, 100977, 96301, 60016, 53296, 24076, 69983, 48602, 46849, 18071, 40076, 68543, 24282, 58426, 546, 92425, 32519, 26982, 56346, 26478, 6318, 42793, 90921, 74446, 10039, 99355, 83968, 7652, 31572, 60024, 18701, 8450, 48842, 1481, 91273, 36751, 23526, 3248, 78821, 69445, 79255, 39675, 4823, 77744, 38521, 22373, 94021, 28502, 46647, 55175, 18503, 33562, 82179, 42919, 92150, 4340, 76218, 34353, 7568, 24102, 6976, 20454, 11297, 98001, 17002, 75227, 82846, 64377, 102378, 31425, 18357, 68923, 95178, 39683, 68241, 76475, 18245, 27899, 81830, 65674, 84370, 52880, 32346, 88175, 85643, 92454, 66331, 85992, 80496, 69277, 93729, 64673, 91179, 92730, 93201, 33033, 72450, 17756, 58725, 63146, 35628, 61403, 64379, 17927, 76644, 92321, 95494, 73969, 81698, 18022, 13679, 77528, 66916, 81019, 21257, 37510, 90020, 8718, 85225, 87075, 98171, 83641, 99921, 25348, 3338, 79921, 90671, 3836, 90835, 90821, 38670, 63570, 54103, 92839, 94320, 41897, 27447, 81322, 52874, 70148, 57999, 71172, 53359, 43952, 37748, 83178, 90500, 69936, 31569, 23296, 853, 38377, 89627, 68275, 38693, 83700, 50578, 32919, 3919, 15247, 56469, 37773, 9379, 49027, 32464, 71867, 1399, 45029, 55247, 33257, 67426, 19047, 51047, 55427, 9907, 21879, 100501, 45594, 101669, 3132, 97675, 5807, 34656, 71949, 6619, 42222, 100417, 59721, 17349, 25274, 83967, 47879, 83856, 18147, 99028, 1718, 53249, 15795, 33399, 13075, 87220, 57679, 37874, 31666, 68490, 40796, 93594, 4043, 70802, 5453, 26522, 56800, 3920, 19025, 29550, 35032, 81298, 40630, 6573, 7490, 75342, 48627, 77868, 16432, 28167, 20825, 52820, 81018, 72176, 89609, 24472, 25427, 51518, 62424, 20050, 93022, 89853, 95153, 96582, 66951, 7144, 68174, 96125, 79100, 8703, 2771, 76506, 28826, 84765, 80999, 91803, 92442, 87995, 78725, 59949, 56580, 82518, 66701, 27894, 8171, 47958, 3693, 60875, 18218, 100349, 1326, 56020, 20430, 53397, 86526, 96695, 1803, 79779, 850, 79373, 57268, 71117, 47563, 8225, 78003, 74900, 72672, 81703, 99627, 46794, 33974, 73751, 14665, 51942, 7746, 38681, 40223, 35206, 101551, 71675, 51545, 41376, 51592, 86380, 68204, 16749, 10075, 56798, 63307, 104, 51305, 75423, 12021, 2306, 32401, 87900, 73050, 91930, 86545, 35770, 93271, 7553, 27219, 53371, 25974, 10789, 48147, 88821, 67728, 18899, 58753, 42524, 96949, 61198, 34247, 44776, 27373, 53778, 15020, 556, 24061, 2653, 57858, 65751, 98176, 20229, 63382, 935, 71294, 41839, 43477, 50477, 48503, 77051, 33480, 53074, 61572, 57393, 5218, 98336, 72631, 2150, 978, 24217, 76051, 31773, 47058, 49129, 31328, 38655, 41651, 16877, 15777, 99723, 14995, 95554, 3169, 83377, 21129, 66097, 4150, 2481, 51579, 26848, 43479, 81876, 8574, 51303, 61248, 27976, 74690, 4372, 27895, 4528, 55998, 75019, 6522, 20474, 101850, 27424, 55484, 8088, 33794, 14504, 64646, 22949, 28451, 49420, 93798, 99750, 77728, 17021, 31302, 5173, 49171, 42329, 12721, 15446, 45072, 67301, 81940, 26126, 15068, 67121, 92226, 34264, 60077, 39051, 23731, 44420, 20544, 93997, 94053, 38117, 97952, 91744, 87077, 83779, 65078, 8924, 92374, 38855, 58204, 59180, 97896, 15334, 49859, 9810, 98703, 82146, 28167, 40897, 51298, 100053, 7900, 83585, 94648, 75049, 66318, 73077, 69620, 24748, 18973, 15082, 47022, 6845, 81022, 31219, 5628, 13865, 30611, 58568, 35171, 51076, 26160, 55768, 84354, 46267, 11311, 63529, 70679, 53156, 49073, 84356, 53856, 3671, 69898, 27454, 10822, 46939, 26226, 85510, 66172, 72522, 60873, 101346, 62673, 7877, 93199, 78474, 78395, 17458, 45677, 59983, 4030, 5273, 101051, 55871, 13102, 21544, 44572, 42486, 72477, 12228, 65493, 84701, 39251, 49077, 97075, 57068, 45605, 90032, 8900, 39874, 40327, 32475, 7004, 100777, 99676, 22747, 11467, 77495, 65567, 68679, 34422, 55224, 36129, 29639, 25158, 18778, 64075, 60897, 29196, 54745, 78911, 98796, 6704, 10682, 97701, 75331, 74305, 22456, 65869, 3810, 94178, 95995, 35175, 97074, 30023, 82946, 81236, 4554, 7201, 75046, 46401, 4371, 32133, 97535, 84154, 4026, 67531, 32798, 5497, 34650, 7075, 78829, 73449, 53122, 28222, 54597, 33155, 91123, 2399, 79444, 15280, 25726, 49731, 46625, 45945, 63832, 101553, 25856, 46507, 4528, 101840, 44144, 43846, 46612, 20274, 68648, 63074, 98304, 66486, 41301, 33827, 40327, 91802, 80404, 8026, 25181, 36948, 18309, 91770, 91672, 46107, 102129, 35816, 11784, 36618, 29951, 22698, 87923, 21296, 61026, 13851, 91076, 98724, 84323, 51751, 4852, 97403, 6966, 80613, 3532, 48808, 84369, 93493, 87274, 71651, 88448, 46497, 13122, 60619, 17547, 40226, 13154, 11877, 93620, 81619, 1250, 85857, 72857, 55501, 97799, 43330, 24475, 26143, 88096, 65452, 21728, 80590, 80025, 82914, 83068, 83954, 36424, 48651, 99689, 17804, 42273, 23948, 41675, 80029, 73351, 1019, 63062, 10689, 17072, 62152, 33601, 48731, 22447, 44370, 50504, 61572, 87772, 82737, 53070, 99850, 61372, 90168, 16031, 79852, 90744, 68583, 67930, 22869, 17442, 1722, 75001, 11076, 26976, 17814, 92226, 76624, 8642, 7075, 80851, 95683, 59174, 44563, 75642, 67823, 84183, 62347, 93026, 38442, 54139, 27153, 30477, 50726, 76570, 50772, 87945, 59097, 22074, 32645, 70524, 28744, 63754, 25224, 26472, 40895, 58627, 36448, 80466, 58579, 99753, 90026, 66800, 44025, 67751, 60552, 93650, 83098, 2903, 60774, 69202, 49374, 829, 3249, 44351, 28351, 7097, 11842, 42468, 63999, 85921, 36748, 40833, 27155, 64948, 44278, 13608, 5569, 50996, 78425, 25553, 1276, 93154, 60405, 95431, 50878, 58270, 35666, 78932, 63745, 16105, 96096, 16959, 62497, 71040, 99672, 71925, 63390, 21026, 38348, 43599, 29598, 15404, 66921, 57578, 12473, 39219, 50073, 1729, 44224, 45556, 8870, 99008, 98404, 81702, 24713, 61106, 85229, 93427, 2037, 77645, 19420, 46739, 16402, 81678, 24743, 48598, 71253, 22651, 56329, 101226, 55629, 52657, 2449, 30109, 62248, 83939, 29479, 88171, 14836, 60851, 102422, 64570, 16812, 93250, 38826, 1873, 67617, 16156, 24550, 85271, 95749, 69247, 33978, 35719, 35028, 91805, 76871, 14532, 53066, 52270, 98222, 56029, 95738, 30400, 43503, 26219, 472, 8975, 20670, 33545, 2304, 15604, 94697, 5851, 25917, 5155, 102023, 11291, 93597, 21171, 65675, 90430, 48598, 29703, 89646, 25588, 29543, 8553, 46884, 58941, 99727, 65669, 21923, 77622, 66198, 33169, 31847, 33252, 66850, 4110, 58981, 76725, 50947, 93095, 82536, 38554, 46426, 84697, 58700, 72164 };

    int *d_x;
    int *d_y;

    // Block, Grid and Shared Memory size
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    int shared_mem_size = block_size * 2 * sizeof(int);  // Shared memory size



    // Calculate means
    // GPU memory
    int *d_x_mean, *d_y_mean;
    gpuErrchk( cudaMalloc((void**)&d_x, N * sizeof(int)) )
    gpuErrchk( cudaMalloc((void**)&d_y, N * sizeof(int)) );
    gpuErrchk( cudaMalloc((void**)&d_x_mean, sizeof(int)) );
    gpuErrchk( cudaMalloc((void**)&d_y_mean, sizeof(int)) );
    gpuErrchk( cudaMemcpy(d_x, h_x, N * sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_y, h_y, N * sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemset(d_x_mean, 0, sizeof(int)) );
    gpuErrchk( cudaMemset(d_y_mean, 0, sizeof(int)) );

    // Setup Timer
    float time;
    cudaEvent_t start, stop;
    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );
    gpuErrchk( cudaEventRecord(start, 0) );

    // Calculate means kernel
    calculatePartialSums<<<grid_size, block_size, shared_mem_size>>>(d_x, d_y, d_x_mean, d_y_mean, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    int x_mean, y_mean;
    gpuErrchk( cudaMemcpy(&x_mean, d_x_mean, sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(&y_mean, d_y_mean, sizeof(int), cudaMemcpyDeviceToHost) );
    x_mean = x_mean / N;
    y_mean = y_mean / N;

    // End Timer
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
    gpuErrchk( cudaEventElapsedTime(&time, start, stop) );
    printf(" -- calculatePartialSums time:  %f ms \n\n", time);

    // GPU Cleanup
    cudaFree(d_x_mean);
    cudaFree(d_y_mean);



    // Calculate coefficients
    // GPU memory
    float *d_num;
    float *d_den;
    gpuErrchk( cudaMalloc((void**)&d_num, sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_den, sizeof(float)) );
    gpuErrchk( cudaMemset(d_num, 0.0f, sizeof(float)) );
    gpuErrchk( cudaMemset(d_den, 0.0f, sizeof(float)) );

    // Setup Timer (only need to record)
    gpuErrchk( cudaEventRecord(start, 0) );

    // Calculates coefficients kernel
    calculatePartialCoefficients<<<grid_size, block_size, shared_mem_size>>>(d_x, d_y, x_mean, y_mean, d_num, d_den, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    float numerator, denominator;
    gpuErrchk( cudaMemcpy(&numerator, d_num, sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(&denominator, d_den, sizeof(float), cudaMemcpyDeviceToHost) );

    // Calculate slope and bias
    float slope, bias;
    slope = numerator / denominator;
    bias = y_mean - slope * x_mean;
    printf ("slope %f  bias %f\n\n", slope, bias);

    // End Timer
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
    gpuErrchk( cudaEventElapsedTime(&time, start, stop) );
    printf(" -- calculatePartialCoefficients time:  %f ms \n\n", time);

    // GPU Cleanup
    cudaFree(d_num);
    cudaFree(d_den);



    // Make Predictions
    // GPU memory
    int *d_predictions, h_predictions[N];
    gpuErrchk( cudaMalloc((void**)&d_predictions, N * sizeof(int)) );

    // Setup Timer (only need to record)
    gpuErrchk( cudaEventRecord(start, 0) );

    // Run predictions kernel
    makePredictions<<<grid_size, block_size>>>(d_x, d_predictions, slope, bias, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Copy data from GPU and print
    gpuErrchk( cudaMemcpy(h_predictions, d_predictions, N*sizeof(int), cudaMemcpyDeviceToHost) );
    printf("Predictions:\n");
    for (int i = 0; i < 10; i++) {
      printf ("%i : %i\n", h_x[i], h_predictions[i]);
    }
    printf ("\n");

    // End Timer
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
    gpuErrchk( cudaEventElapsedTime(&time, start, stop) );
    printf(" -- makePredictions time:  %f ms \n\n", time);

    // GPU Cleanup
    cudaFree(d_x);


    // Calculate MSE
    // GPU Memory
    float* d_mse;
    float mse;

    // Allocate and initialize the MSE variable on the device
    gpuErrchk( cudaMalloc((void**)&d_mse, sizeof(float)) );

    // Setup Timer (only need to record)
    gpuErrchk( cudaEventRecord(start, 0) );

    // Run the kernel
    calculatePartialMSE<<<grid_size, block_size, shared_mem_size>>>(d_y, d_predictions, d_mse, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Copy the result back to host
    gpuErrchk( cudaMemcpy(&mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost) );
    // Final MSE calculation on the host
    mse = mse / N;
    printf ("MSE: %f\n\n", mse);

    // End Timer
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
    gpuErrchk( cudaEventElapsedTime(&time, start, stop) );
    printf(" -- calculatePartialMSE time:  %f ms \n\n", time);



    // Free GPU memory
    cudaFree(d_mse);
    cudaFree(d_predictions);
    cudaFree(d_y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    return 0;
}