#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#if USE_PNG
#include <png.h>
#endif

#define SWAP(_T, __lhs, __rhs) \
    do { \
        _T __tmp = __lhs; \
        __lhs = __rhs; \
        __rhs = __tmp; \
    } while (0)

#define DATA_DIR "data/"

char *progname;

// utilities {{{

_Noreturn static void error(const char *str) {
    fprintf(stderr, "%s: %s\n", progname, str);
    exit(1);
}

#if RAND_MAX != 0x7fffffff
#undef RAND_MAX
#define RAND_MAX 0x7fffffff
#endif
#define XS128(p) __xorshift128_##p
static uint32_t XS128(x) = 123456789;
static uint32_t XS128(y) = 362436069;
static uint32_t XS128(z) = 521288629;
static uint32_t XS128(w) = 88675123;

void xorshift128_seed(unsigned s) {
    if (s != 0) {
        XS128(w) += s - 1;
    }
}

int xorshift128(void) {
    uint32_t t = XS128(x) ^ (XS128(x) << 11);
    XS128(x) = XS128(y);
    XS128(y) = XS128(z);
    XS128(z) = XS128(w);
    XS128(w) = (XS128(w) ^ (XS128(w) >> 19)) ^ (t ^ (t >> 8));
    return XS128(w) & RAND_MAX;
}
#undef XS128

#define srand xorshift128_seed
#define rand xorshift128

/// convert between big and little endians
int32_t convert_endian_i32(int32_t n) {
    union {
        int32_t __n;
        int8_t __b[4];
    } __u = {.__n = n};
    SWAP(int8_t, __u.__b[0], __u.__b[3]);
    SWAP(int8_t, __u.__b[1], __u.__b[2]);
    return __u.__n;
}

/// return uniform random number [0,1]
static inline double random_uniform(void) {
    return rand() / (double)RAND_MAX;
}

/// return N(0,1) random number
double random_normal(void) {
    static const double pi = 3.141592653589793238;
    double x;
    while ((x = random_uniform()) == 0) {
    }
    double y = random_uniform();
    return sqrt(-2 * log(x)) * cos(2 * pi * y);
}

void random_shuffle(int n, float **x, int8_t *y) {
    for (int i = 0; i < n; ++i) {
        int idx = rand() / (RAND_MAX + 1.0) * n;
        SWAP(float *, x[i], x[idx]);
        SWAP(int8_t, y[i], y[idx]);
    }
}

static inline double crossentropy(float *y, int8_t t) {
    return -log(y[t] + 1e-7);
}

// blas {{{

/// X = alpha*X
void my_sscal(int N, float alpha, float *X) {
    for (int i = 0; i < N; ++i) {
        X[i] *= alpha;
    }
}

/// index of max abs value
int my_isamax(int N, const float *X) {
    int ret = 0;
    for (int i = 1; i < N; ++i) {
        if (fabs(X[ret]) < fabs(X[i])) {
            ret = i;
        }
    }
    return ret;
}

/// Y = alpha*X + Y
void my_saxpy(int n, float alpha, const float *X, float *Y) {
    for (int i = 0; i < n; ++i) {
        // Y[i] += alpha * X[i];
        Y[i] = fma(alpha, X[i], Y[i]);
    }
}

/// dot product
float my_sdot(int n, const float *X, const float *Y) {
    float ret = 0;
    for (int i = 0; i < n; ++i) {
        ret = fmaf(X[i], Y[i], ret);
    }
    return ret;
}

/// y = alpha*A*x + beta*y
void my_sgemv(int M, int N, float alpha, const float *A, const float *x, float beta, float *y) {
    my_sscal(M, beta, y);
    for (int m = 0; m < M; ++m) {
        // for (int n = 0; n < N; ++n) {
        //     y[m] += alpha * A[m * N + n] * x[n];
        // }
        y[m] = fmaf(alpha, my_sdot(N, A + m * N, x), y[m]);
    }
}

/// A = alpha*X*Y^t + A
void my_sger(int M, int N, float alpha, const float *X, const float *Y, float *A) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            A[m * N + n] = fmaf(alpha, X[m] * Y[n], A[m * N + n]);
        }
    }
}

/// C = alpha*A*B + beta*C; A: R^{M*K}; B: R^{K*N}; C: R^{M*N}
void my_sgemm(
  int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    my_sscal(M * N, beta, C);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                C[m * N + n] = fmaf(alpha, A[m * K + k] * B[k * N + n], C[m * N + n]);
            }
        }
    }
}

// }}} blas

// }}} utilities

// BMP {{{

#pragma pack(1)
typedef struct BMP_Header {
    int16_t magic; // "BM"
    int32_t size;
    int16_t reserved1;
    int16_t reserved2;
    int32_t offset;
} BMP_Header;

#pragma pack(1)
typedef struct BMP_InfoHeader {
    int32_t header_size; // always 40
    int32_t width;
    int32_t height;
    int16_t planes; // always 1
    int16_t bit_count;
    int32_t compression;
    int32_t image_size;
    int32_t x_resolusion;
    int32_t y_resolusion;
    int32_t colors;
    int32_t important_colors;
} BMP_InfoHeader;

typedef struct BMP_ColorPallet {
    uint32_t colors[256];
    size_t size;
} BMP_ColorPallet;

/// supports gray-scale only
/// TODO: full support
void BMP_init(BMP_Header *header, BMP_InfoHeader *info, BMP_ColorPallet *pallet) {
    union {
        uint8_t c[2];
        uint16_t s;
    } u = {.c = {'B', 'M'}};

    header->magic = u.s;
    header->size = 0;
    header->reserved1 = header->reserved2 = 0;
    header->offset = 14 + 40 + 4 * (1u << 8);

    info->header_size = sizeof(*info);
    info->planes = 1;
    info->bit_count = 8;
    info->compression = 0;
    info->image_size = 0;
    info->x_resolusion = 0;
    info->y_resolusion = 0;
    info->colors = 0;
    info->important_colors = 0;

    pallet->size = 1u << info->bit_count;
    for (size_t i = 0; i < pallet->size; ++i) {
        pallet->colors[i] = (i << 16) | (i << 8) | i;
    }
}

// }}} BMP

// mnist {{{

#define MNIST_DATA_DIR DATA_DIR "mnist/"
#define MNIST_IMAGE_DIR DATA_DIR "images/"
#define MNIST_TRAIN_IMAGES MNIST_DATA_DIR "train-images-idx3-ubyte"
#define MNIST_TRAIN_LABELS MNIST_DATA_DIR "train-labels-idx1-ubyte"
#define MNIST_TEST_IMAGES MNIST_DATA_DIR "t10k-images-idx3-ubyte"
#define MNIST_TEST_LABELS MNIST_DATA_DIR "t10k-labels-idx1-ubyte"

#define MNIST_TRAIN_COUNT 60000
#define MNIST_TEST_COUNT 10000

#define MNIST_IMAGE_ROWS 28
#define MNIST_IMAGE_COLS 28
#define MNIST_IMAGE_SIZE (MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS)

static void mnist_load_image(float *image, int rows, int cols, FILE *fp) {
    int size = rows * cols;
    uint8_t *tmp = malloc(size * sizeof(*tmp));
    fread(tmp, sizeof(*tmp), size, fp);
    for (int i = 0; i < size; ++i) {
        image[i] = tmp[i] / 256.0;
    }
    free(tmp);
}

static void mnist_load_label(int8_t *label, FILE *fp) {
    fread(label, sizeof(*label), 1, fp);
}

void mnist_init(
  float **train_x, int8_t *train_y, int *train_count, float **test_x, int8_t *test_y,
  int *test_count, int *rows, int *cols) {
    FILE *train_image;
    FILE *train_label;
    FILE *test_image;
    FILE *test_label;

    train_image = fopen(MNIST_TRAIN_IMAGES, "rb");
    train_label = fopen(MNIST_TRAIN_LABELS, "rb");
    test_image = fopen(MNIST_TEST_IMAGES, "rb");
    test_label = fopen(MNIST_TEST_LABELS, "rb");
    if (!train_image || !train_label || !test_image || !test_label) {
        perror("MNIST");
        exit(1);
    }

    fseek(train_image, 4, SEEK_SET);
    fseek(train_label, 4, SEEK_SET);
    fseek(test_image, 4, SEEK_SET);
    fseek(test_label, 4, SEEK_SET);

#define GET_COUNT(name, assume) \
    do { \
        int32_t __tmp; \
        fread(&__tmp, sizeof(__tmp), 1, name##_image); \
        *name##_count = convert_endian_i32(__tmp); \
        fread(&__tmp, sizeof(__tmp), 1, name##_label); \
        assert(*name##_count == convert_endian_i32(__tmp)); \
        assert(*name##_count == assume); \
    } while (0)
    GET_COUNT(train, MNIST_TRAIN_COUNT);
    GET_COUNT(test, MNIST_TEST_COUNT);
#undef GET_COUNT

#define GET_IMAGE_SIZE(name, a) \
    do { \
        int32_t __tmp; \
        fread(&__tmp, sizeof(__tmp), 1, train_image); \
        *name = convert_endian_i32(__tmp); \
        fread(&__tmp, sizeof(__tmp), 1, test_image); \
        assert(*name == convert_endian_i32(__tmp)); \
        assert(*name == a); \
    } while (0)
    GET_IMAGE_SIZE(rows, MNIST_IMAGE_ROWS);
    GET_IMAGE_SIZE(cols, MNIST_IMAGE_COLS);
#undef GET_IMAGE_SIZE

    int size = (*rows) * (*cols);
#define GET_DATA(name) \
    do { \
        for (int i = 0; i < *name##_count; ++i) { \
            name##_x[i] = malloc(size * sizeof(float)); \
            mnist_load_image(name##_x[i], *rows, *cols, name##_image); \
            mnist_load_label(&name##_y[i], name##_label); \
        } \
    } while (0)
    GET_DATA(train);
    GET_DATA(test);
#undef GET_DATA

    fclose(train_image);
    fclose(train_label);
    fclose(test_image);
    fclose(test_label);
}

void mnist_filter(int rows, int cols, float *data) {
    int top = rows;
    int bottom = 0;
    int left = cols;
    int right = 0;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i * cols + j] = data[i * cols + j] < 0.3 ? 0 : 1;
            if (data[i * cols + j] == 0) {
                continue;
            }
            if (top > i) {
                top = i;
            }
            if (bottom < i) {
                bottom = i;
            }
            if (left > j) {
                left = j;
            }
            if (right < j) {
                right = j;
            }
        }
    }
    assert(top < bottom);
    assert(left < right);

    int dr = ((rows - bottom - 1) - top) / 2;
    int dc = ((cols - right - 1) - left) / 2;

    int is_neg = dr < 0 ? 1 : 0;
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            int idx = is_neg ? i : rows - i - 1;
            int move_to = idx + dr;
            if (move_to >= rows || move_to < 0) {
                continue;
            }
            data[move_to * cols + j] = data[idx * cols + j];
        }
    }

    is_neg = dc < 0 ? 1 : 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = is_neg ? j : cols - j - 1;
            int move_to = idx + dc;
            if (move_to >= cols || move_to < 0) {
                continue;
            }
            data[i * cols + move_to] = data[i * cols + idx];
        }
    }
}

void mnist_save_bmp(int32_t width, int32_t height, const float *x, const char *fname) {
    FILE *fp = fopen(fname, "wb");
    if (!fp) {
        perror("MNIST");
        exit(1);
    }

    uint8_t *data = malloc(sizeof(*data) * width * height);
    for (int64_t j = 0; j < width * height; ++j) {
        data[j] = x[j] * 256;
    }

    BMP_Header header;
    BMP_InfoHeader info;
    BMP_ColorPallet pallet;
    BMP_init(&header, &info, &pallet);

    info.width = width;
    info.height = height;

    fwrite(&header, 14, 1, fp);
    fwrite(&info, 40, 1, fp);
    fwrite(&pallet.colors, sizeof(pallet.colors[0]), pallet.size, fp);
    for (int32_t i = height; i > 0; --i) {
        for (int32_t j = 0; j < width; ++j) {
            fwrite(&data[(i - 1) * width + j], sizeof(*data), 1, fp);
        }
        if (width & 3) {
            uint8_t __buf[3] = {0};
            fwrite(__buf, sizeof(uint8_t), 4 - (width & 3), fp);
        }
    }

    free(data);
    fclose(fp);
}

void mnist_load_bmp(float *x, const char *fname) {
    FILE *fp = fopen(fname, "rb");
    if (!fp) {
        perror("MNIST");
        exit(1);
    }

    BMP_Header header;
    BMP_InfoHeader info;
    BMP_ColorPallet pallet;
    fread(&header, sizeof(header), 1, fp);
    fread(&info, sizeof(info), 1, fp);
    fread(pallet.colors, sizeof(pallet.colors[0]), 256, fp);
    if (
      info.header_size != 40 || info.bit_count != 8 || info.width != MNIST_IMAGE_COLS
      || info.height != MNIST_IMAGE_ROWS) {
        error("Unsupported file type");
    }

    uint8_t tmp;
    for (int32_t i = info.height; i > 0; --i) {
        for (int32_t j = 0; j < info.width; ++j) {
            fread(&tmp, sizeof(tmp), 1, fp);
            x[(i - 1) * info.width + j] = tmp / 256.0;
        }
        if (info.width & 3) {
            uint8_t __buf[3];
            fread(__buf, sizeof(uint8_t), 4 - (info.width & 3), fp);
        }
    }
    fclose(fp);
}

#if USE_PNG
void mnist_save_png(int width, int height, float *x, const char *fname) {
    FILE *fp = fopen(fname, "wb");
    if (!fp) {
        perror("MNIST");
        exit(1);
    }

    uint8_t *data = malloc(sizeof(*data) * height * width);
    for (int j = 0; j < height * width; ++j) {
        data[j] = x[j] * 256;
    }

    png_structp png;
    png_infop info;
    png_bytepp datap;

    png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    info = png_create_info_struct(png);

    png_init_io(png, fp);
    png_set_IHDR(
      png, info, width, height, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    datap = png_malloc(png, sizeof(png_bytep) * height);
    png_set_rows(png, info, datap);

    for (int j = 0; j < height; ++j) {
        datap[j] = png_malloc(png, sizeof(png_byte) * width);
        memcpy(datap[j], data + j * width, width);
    }
    png_write_png(png, info, PNG_TRANSFORM_IDENTITY, NULL);

    for (int j = 0; j < height; ++j) {
        png_free(png, datap[j]);
    }

    png_free(png, datap);
    png_destroy_write_struct(&png, &info);
    free(data);
    fclose(fp);
}

void mnist_load_png(float *x, const char *fname) {
    FILE *fp = fopen(fname, "rb");
    if (!fp) {
        perror("MNIST");
        exit(1);
    }

    png_structp png;
    png_infop info;
    png_bytepp datap;
    png_byte type;
    png_byte header[8];
    int width;
    int height;

    fread(header, sizeof(header[0]), 8, fp);

    if (png_sig_cmp(header, 0, 8)) {
        error("Unsupported file type");
    }

    png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        perror("MNIST");
        exit(1);
    }

    info = png_create_info_struct(png);
    if (!png) {
        perror("MNIST");
        exit(1);
    }

    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);
    png_read_png(png, info, PNG_TRANSFORM_PACKING | PNG_TRANSFORM_STRIP_16, NULL);

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    datap = png_get_rows(png, info);
    type = png_get_color_type(png, info);
    if (type != PNG_COLOR_TYPE_GRAY) {
        error("Unsupported file type");
    }

    uint8_t *data = malloc(width * height * sizeof(*data));
    for (int i = 0; i < height; i++) {
        memcpy(data + i * width, datap[i], width);
    }
    for (int i = 0; i < width * height; ++i) {
        x[i] = data[i] / 256.0;
    }

    png_destroy_read_struct(&png, &info, NULL);
    free(data);
    fclose(fp);
}
#endif

void mnist_save_images(void) {
    float **train_x;
    int8_t *train_y;
    int train_count;
    float **test_x;
    int8_t *test_y;
    int test_count;
    int rows;
    int cols;

    train_x = malloc(MNIST_TRAIN_COUNT * sizeof(float *));
    train_y = malloc(MNIST_TRAIN_COUNT * sizeof(int8_t));
    test_x = malloc(MNIST_TEST_COUNT * sizeof(float *));
    test_y = malloc(MNIST_TEST_COUNT * sizeof(int8_t));
    mnist_init(train_x, train_y, &train_count, test_x, test_y, &test_count, &rows, &cols);

    struct stat s;
    if (stat(MNIST_IMAGE_DIR, &s) != 0) {
#if defined(_WIN32) || defined(_WIN64)
        mkdir(MNIST_IMAGE_DIR);
#else
        mkdir(MNIST_IMAGE_DIR, 0755);
#endif
    }

#if USE_PNG
#define IMAGE_EXT ".png"
#define save_image mnist_save_png
#else
#define IMAGE_EXT ".bmp"
#define save_image mnist_save_bmp
#endif

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < train_count; ++i) {
        char fname[32];
        snprintf(fname, sizeof(fname), MNIST_IMAGE_DIR "train-%05d" IMAGE_EXT, i);
        save_image(cols, rows, train_x[i], fname);
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < test_count; ++i) {
        char fname[32];
        snprintf(fname, sizeof(fname), MNIST_IMAGE_DIR "test-%05d" IMAGE_EXT, i);
        save_image(cols, rows, test_x[i], fname);
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < train_count; ++i) {
        free(train_x[i]);
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < test_count; ++i) {
        free(test_x[i]);
    }
    free(train_x);
    free(train_y);
    free(test_x);
    free(test_y);
}

// }}} mnist

#define PARAMS_DIR DATA_DIR "params/"
#define PARAM_FILE(s) PARAMS_DIR #s

#define FC_LAYERS 3
#define LAYERS (2 * FC_LAYERS)
#define FC0_ROWS 50
#define FC0_COLS MNIST_IMAGE_SIZE
#define FC0_SIZE (FC0_ROWS * FC0_COLS)
#define FC1_ROWS 100
#define FC1_COLS FC0_ROWS
#define FC1_SIZE (FC1_ROWS * FC1_COLS)
#define FC2_ROWS 10
#define FC2_COLS FC1_ROWS
#define FC2_SIZE (FC2_ROWS * FC2_COLS)

#define BATCH_SIZE 100

typedef enum Mode {
    kTrain,
    kInference,
    kSavePictures,
} Mode;

typedef enum FileType {
    BMP,
#if USE_PNG
    PNG,
#endif
} FileType;

static float *fc_W[FC_LAYERS];
static float *fc_b[FC_LAYERS];
static float *fc_dW[FC_LAYERS][BATCH_SIZE];
static float *fc_db[FC_LAYERS][BATCH_SIZE];
static float *momentum_W[FC_LAYERS];
static float *momentum_b[FC_LAYERS];
static const int fc_W_size[FC_LAYERS] = {FC0_SIZE, FC1_SIZE, FC2_SIZE};
static const int fc_b_size[FC_LAYERS] = {FC0_ROWS, FC1_ROWS, FC2_ROWS};
static struct NetworkDim {
    int in;
    int out;
} const dims[LAYERS] = {{FC0_COLS, FC0_ROWS}, {FC0_ROWS, FC0_ROWS}, {FC1_COLS, FC1_ROWS},
                        {FC1_ROWS, FC1_ROWS}, {FC2_COLS, FC2_ROWS}, {FC2_ROWS, FC2_ROWS}};

static void initialize(Mode mode) {
    if (mode == kSavePictures) {
        return;
    }
    for (int i = 0; i < FC_LAYERS; ++i) {
        fc_W[i] = malloc(fc_W_size[i] * sizeof(float));
        fc_b[i] = malloc(fc_b_size[i] * sizeof(float));
        if (mode == kTrain) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int j = 0; j < BATCH_SIZE; ++j) {
                fc_dW[i][j] = malloc(fc_W_size[i] * sizeof(float));
                fc_db[i][j] = malloc(fc_b_size[i] * sizeof(float));
            }
            momentum_W[i] = calloc(fc_W_size[i], sizeof(float));
            momentum_b[i] = calloc(fc_b_size[i], sizeof(float));
        }
    }
}

static void finalize(Mode mode) {
    for (int i = 0; i < FC_LAYERS; ++i) {
        free(fc_W[i]);
        free(fc_b[i]);
        if (mode == kTrain) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int j = 0; j < BATCH_SIZE; ++j) {
                free(fc_dW[i][j]);
                free(fc_db[i][j]);
            }
            free(momentum_W[i]);
            free(momentum_b[i]);
        }
    }
}

static void load_param(float *dest, int len, const char *fname) {
    char str[32];
    snprintf(str, sizeof(str), PARAMS_DIR "%s", fname);
    FILE *fp = fopen(str, "rb");
    if (!fp) {
        for (int i = 0; i < len; ++i) {
            // dest[i] = random_uniform() * 2 - 1;
            dest[i] = 0.5 * random_normal();
        }
    } else {
        fread(dest, sizeof(float), len, fp);
        fclose(fp);
    }
}

static void save_param(const float *src, int len, const char *fname) {
    struct stat s;
    if (stat(PARAMS_DIR, &s) != 0) {
#if defined(_WIN32) || defined(_WIN64)
        mkdir(PARAMS_DIR);
#else
        mkdir(PARAMS_DIR, 0755);
#endif
    }

    char str[32];
    snprintf(str, sizeof(str), PARAMS_DIR "%s", fname);
    FILE *fp = fopen(str, "wb");
    if (!fp) {
        perror(progname);
        exit(1);
    }
    fwrite(src, sizeof(float), len, fp);
    fclose(fp);
}

// network {{{

/// y = W*x + b
void affine(
  int m, int n, const float *restrict x, const float *restrict W, const float *restrict b,
  float *restrict y) {
    memcpy(y, b, m * sizeof(float));
    my_sgemv(m, n, 1, W, x, 1, y);
}

void relu(int n, const float *restrict x, float *restrict y) {
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] > 0 ? x[i] : 0;
    }
}

void softmax(int n, const float *restrict x, float *restrict y) {
    int idx = my_isamax(n, x);
    double tmp = 0;
    for (int i = 0; i < n; ++i) {
        tmp += exp(x[i] - x[idx]);
    }
    for (int i = 0; i < n; ++i) {
        y[i] = exp(x[i] - x[idx]) / tmp;
    }
}

void softmax_with_loss_bwd(int n, const float *restrict y, int8_t t, float *restrict dx) {
    for (int i = 0; i < n; ++i) {
        dx[i] = y[i] - (t == i ? 1 : 0);
    }
}

void relu_bwd(int n, const float *restrict x, const float *restrict dy, float *restrict dx) {
    for (int i = 0; i < n; ++i) {
        dx[i] = x[i] > 0 ? dy[i] : 0;
    }
}

void affine_bwd(
  int m, int n, const float *restrict x, const float *restrict W, const float *restrict dy,
  float *restrict dW, float *restrict db, float *restrict dx) {
    // dW = dy * x^{T}
    memset(dW, 0, m * n * sizeof(float));
    my_sger(m, n, 1, dy, x, dW);

    // db = dy
    memcpy(db, dy, m * sizeof(float));

    // dx = W^{T} * dy
    memset(dx, 0, n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            // dx[i] += W[j * n + i] * dy[j];
            dx[i] = fmaf(W[j * n + i], dy[j], dx[i]);
        }
    }
}

// }}} network

static void load_params(void) {
    static const char *s[FC_LAYERS * 2] = {"fc0_W", "fc0_b", "fc1_W", "fc1_b", "fc2_W", "fc2_b"};
    for (int i = 0; i < FC_LAYERS; ++i) {
        load_param(fc_W[i], fc_W_size[i], s[i * 2]);
        load_param(fc_b[i], fc_b_size[i], s[i * 2 + 1]);
    }
}

static void save_params(void) {
    static const char *s[FC_LAYERS * 2] = {"fc0_W", "fc0_b", "fc1_W", "fc1_b", "fc2_W", "fc2_b"};
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < FC_LAYERS; ++i) {
        save_param(fc_W[i], fc_W_size[i], s[i * 2]);
        save_param(fc_b[i], fc_b_size[i], s[i * 2 + 1]);
    }
}

typedef struct Result {
    int idx;
    int count;
    double loss;
} Result;

static Result minibatch_train(float **x, const int8_t *t, float eta, float alpha) {
    int c = 0;
    double l = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : c, l)
#endif
    for (int i = 0; i < BATCH_SIZE; ++i) {
        float *z[LAYERS + 1];
        z[0] = x[i];
        for (int j = 0; j < LAYERS; ++j) {
            z[j + 1] = malloc(dims[j].out * sizeof(float));
        }

        affine(dims[0].out, dims[0].in, z[0], fc_W[0], fc_b[0], z[1]);
        relu(dims[1].in, z[1], z[2]);
        affine(dims[2].out, dims[2].in, z[2], fc_W[1], fc_b[1], z[3]);
        relu(dims[3].in, z[3], z[4]);
        affine(dims[4].out, dims[4].in, z[4], fc_W[2], fc_b[2], z[5]);
        softmax(dims[5].in, z[5], z[6]);

        float *y = z[LAYERS];
        int idx = my_isamax(dims[LAYERS - 1].out, y);
        c += idx == t[i];
        l += crossentropy(y, t[i]);

        float *dx[LAYERS];
        for (int j = 0; j < LAYERS; ++j) {
            dx[j] = malloc(dims[j].in * sizeof(float));
        }

        softmax_with_loss_bwd(dims[5].in, z[6], t[i], dx[5]);
        affine_bwd(dims[4].out, dims[4].in, z[4], fc_W[2], dx[5], fc_dW[2][i], fc_db[2][i], dx[4]);
        relu_bwd(dims[3].in, z[3], dx[4], dx[3]);
        affine_bwd(dims[2].out, dims[2].in, z[2], fc_W[1], dx[3], fc_dW[1][i], fc_db[1][i], dx[2]);
        relu_bwd(dims[1].in, z[1], dx[2], dx[1]);
        affine_bwd(dims[0].out, dims[0].in, z[0], fc_W[0], dx[1], fc_dW[0][i], fc_db[0][i], dx[0]);

        for (int j = 0; j < LAYERS; ++j) {
            free(z[j + 1]);
            free(dx[j]);
        }
    }

    float __eta = eta / BATCH_SIZE;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < FC_LAYERS; ++i) {
        float *fc_dW_s = calloc(fc_W_size[i], sizeof(float));
        float *fc_db_s = calloc(fc_b_size[i], sizeof(float));

        for (int j = 0; j < BATCH_SIZE; ++j) {
            my_saxpy(fc_W_size[i], 1, fc_dW[i][j], fc_dW_s);
            my_saxpy(fc_b_size[i], 1, fc_db[i][j], fc_db_s);
        }

        my_sscal(fc_W_size[i], alpha, momentum_W[i]);
        my_saxpy(fc_W_size[i], __eta, fc_dW_s, momentum_W[i]);
        my_saxpy(fc_W_size[i], -1, momentum_W[i], fc_W[i]);
        my_sscal(fc_b_size[i], alpha, momentum_b[i]);
        my_saxpy(fc_b_size[i], __eta, fc_db_s, momentum_b[i]);
        my_saxpy(fc_b_size[i], -1, momentum_b[i], fc_b[i]);

        free(fc_dW_s);
        free(fc_db_s);
    }

    return (Result){-1, c, l};
}

static Result inference(float *x, int8_t t) {
    int c;
    double l;
    float *z[FC_LAYERS * 2 + 1];
    z[0] = x;
    for (int j = 0; j < LAYERS; ++j) {
        z[j + 1] = malloc(dims[j].out * sizeof(float));
    }

    affine(dims[0].out, dims[0].in, z[0], fc_W[0], fc_b[0], z[1]);
    relu(dims[1].in, z[1], z[2]);
    affine(dims[2].out, dims[2].in, z[2], fc_W[1], fc_b[1], z[3]);
    relu(dims[3].in, z[3], z[4]);
    affine(dims[4].out, dims[4].in, z[4], fc_W[2], fc_b[2], z[5]);
    softmax(dims[5].in, z[5], z[6]);

    float *y = z[LAYERS];
    int idx = my_isamax(dims[LAYERS - 1].out, y);
    c = idx == t;
    l = crossentropy(y, t);

    for (int i = 1; i < LAYERS + 1; ++i) {
        free(z[i]);
    }

    return (Result){idx, c, l};
}

void train_main(int epochs, float eta, float decay, float alpha, int seed) {
    float **train_x;
    int8_t *train_y;
    int train_count;
    float **test_x;
    int8_t *test_y;
    int test_count;
    int rows;
    int cols;

    srand(seed);
    for (int i = 0; i < 10; ++i) {
        rand();
    }

    train_x = malloc(MNIST_TRAIN_COUNT * sizeof(float *));
    train_y = malloc(MNIST_TRAIN_COUNT * sizeof(int8_t));
    test_x = malloc(MNIST_TEST_COUNT * sizeof(float *));
    test_y = malloc(MNIST_TEST_COUNT * sizeof(int8_t));
    mnist_init(train_x, train_y, &train_count, test_x, test_y, &test_count, &rows, &cols);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < MNIST_TRAIN_COUNT; ++i) {
        mnist_filter(rows, cols, train_x[i]);
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < MNIST_TEST_COUNT; ++i) {
        mnist_filter(rows, cols, test_x[i]);
    }

    load_params();

    for (int e = 0; e < epochs; ++e) {
        int c = 0;
        double l = 0;
        random_shuffle(train_count, train_x, train_y);
        for (int i = 0; i < MNIST_TRAIN_COUNT / BATCH_SIZE; ++i) {
            Result r = minibatch_train(
              train_x + i * BATCH_SIZE, train_y + i * BATCH_SIZE, eta - decay * e, alpha);
            c += r.count;
            l += r.loss;
        }
        printf(
          "%d/%d: %.4f %.4f", e + 1, epochs, c / (double)MNIST_TRAIN_COUNT, l / MNIST_TRAIN_COUNT);

        c = 0;
        l = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : c, l)
#endif
        for (int i = 0; i < MNIST_TEST_COUNT; ++i) {
            Result r = inference(test_x[i], test_y[i]);
            c += r.count;
            l += r.loss;
        }
        printf(" %.4f %.4f\n", c / (double)MNIST_TEST_COUNT, l / MNIST_TEST_COUNT);
    }

    save_params();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < train_count; ++i) {
        free(train_x[i]);
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < test_count; ++i) {
        free(test_x[i]);
    }
    free(train_x);
    free(train_y);
    free(test_x);
    free(test_y);
}

void inference_main(char *fname) {
    FileType type;
    char *ext = NULL;

    for (int i = strlen(fname); i >= 0; --i) {
        if (fname[i] == '.') {
            ext = &fname[i];
            break;
        }
    }

    if (!ext) {
        fprintf(stderr, "%s: %s: Unsupported file type\n", progname, fname);
        exit(1);
    } else if (!strcmp(ext, ".bmp")) {
        type = BMP;
#if USE_PNG
    } else if (!strcmp(ext, ".png")) {
        type = PNG;
#endif
    } else {
        fprintf(stderr, "%s: %s: Unsupported file type\n", progname, fname);
        exit(1);
    }

    float *x = calloc(MNIST_IMAGE_SIZE, sizeof(float));
    switch (type) {
    case BMP:
        mnist_load_bmp(x, fname);
        break;
#if USE_PNG
    case PNG:
        mnist_load_png(x, fname);
        break;
#endif
    }

    mnist_filter(MNIST_IMAGE_ROWS, MNIST_IMAGE_COLS, x);
    load_params();
    printf("%d\n", inference(x, -1).idx);
}

int print_usage(void) {
    fprintf(
      stderr,
      "Usage: %s [OPTIONS]...\n"
      "\nModes:\n"
      "  -i FILE    inference from FILE;\n"
      "               FILE must be 28*28 Windows Bitmap"
#if USE_PNG
      " or PNG"
#endif
      "\n"
      "  -p         convert datasets to pictures\n"
      "  -t         train (default)\n"
      "\nParameters:\n"
      "  -a NUMBER  momentum (default is 0.9)\n"
      "  -d NUMBER  decay for learning rate (default is 1e-6)\n"
      "  -n NUMBER  number of epochs (default is 10)\n"
      "  -r NUMBER  learning rate (default is 0.01)\n"
      "  -s NUMBER  seed for random number (default is 42)\n"
      "\n"
      "  -h         display this help and exit\n",
      progname);
    return 1;
}

int main(int argc, char **argv) {
    int epochs = 10;
    float eta = 0.01;
    float decay = 1e-6;
    float alpha = 0.9;
    int seed = 42;
    char *fname = NULL;
    Mode mode = kTrain;
    progname = argv[0];

    int c;
    while ((c = getopt(argc, argv, "a:d:i:hn:pr:s:t")) != -1) {
        switch (c) {
        case 'a':
            alpha = strtof(optarg, NULL);
            break;
        case 'd':
            decay = strtof(optarg, NULL);
            break;
        case 'i':
            mode = kInference;
            fname = optarg;
            break;
        case 'n':
            epochs = strtol(optarg, NULL, 10);
            break;
        case 'p':
            mode = kSavePictures;
            break;
        case 'r':
            eta = strtof(optarg, NULL);
            break;
        case 's':
            seed = atoi(optarg);
            break;
        case 't':
            mode = kTrain;
            break;
        default:
            return print_usage();
        }
    }

    initialize(mode);

    switch (mode) {
    case kTrain:
        train_main(epochs, eta, decay, alpha, seed);
        break;
    case kInference:
        inference_main(fname);
        break;
    case kSavePictures:
        mnist_save_images();
    }

    finalize(mode);

    return 0;
}
