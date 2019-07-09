#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#if USE_BLAS
#include <cblas.h>
#endif
#if USE_PNG
#include <png.h>
#endif

/// directory to store data
#define DATA_DIR "data/"

/// program name
const char *progname;

// utilities {{{

_Noreturn static void error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "%s: ", progname);
    vfprintf(stderr, fmt, args);
    exit(1);
}

/// swap the values whose types are `T`
#define SWAP(T, a, b) \
    do { \
        T __tmp = a; \
        a = b; \
        b = __tmp; \
    } while (0)

/// convert between big and little endians
int32_t convert_endian_i32(int32_t n) {
    union {
        int32_t n;
        int8_t b[4];
    } u = {.n = n};
    SWAP(int8_t, u.b[0], u.b[3]);
    SWAP(int8_t, u.b[1], u.b[2]);
    return u.n;
}

// random {{{

/// platform-independent pseudorandom number generator which aims to provide reproducibility
#ifndef RAND_MAX
#define RAND_MAX 0x7fffffff
#elif RAND_MAX != 0x7fffffff
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
        XS128(w) = (88675123 - 1) + s;
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

/// override rand() and srand() provided by C standard library
#define srand xorshift128_seed
#define rand xorshift128

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

// }}} random

// blas {{{

/// X = alpha*X
/// - X: N vector
void myblas_sscal(int N, float alpha, float *X) {
#if USE_BALS
    cblas_sscal(N, alpha, X, 1);
#else
    if (alpha == 1) {
        return;
    }

    for (int i = 0; i < N; ++i) {
        X[i] *= alpha;
    }
#endif
}

/// return index of max value
/// - X: N vector
/// **NOTE**: not provided by BLAS
int myblas_ismax(int N, const float *X) {
    int ix = 0;
    for (int i = 1; i < N; ++i) {
        if (X[ix] < X[i]) {
            ix = i;
        }
    }
    return ix;
}

/// Y = alpha*X + Y
/// - X,Y: N vector
void myblas_saxpy(int N, float alpha, const float *X, float *Y) {
#if USE_BLAS
    cblas_saxpy(N, alpha, X, 1, Y, 1);
#else
    if (alpha == 0) {
        return;
    }

    for (int i = 0; i < N; ++i) {
        Y[i] = fmaf(alpha, X[i], Y[i]);
    }
#endif
}

/// dot product
/// - X,Y: N vector
float myblas_sdot(int N, const float *X, const float *Y) {
#if USE_BLAS
    return cblas_sdot(N, X, 1, Y, 1);
#else
    float sum = 0;
    for (int i = 0; i < N; ++i) {
        sum = fmaf(X[i], Y[i], sum);
    }
    return sum;
#endif
}

/// Y = alpha*A*X + beta*Y;
/// - A: M*N matrix
/// - X: N vector
/// - Y: M vector
void myblas_sgemv(int M, int N, float alpha, const float *A, const float *X, float beta, float *Y) {
#if USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, alpha, A, N, X, 1, beta, Y, 1);
#else
    if (beta != 1) {
        myblas_sscal(M, beta, Y);
    }
    if (alpha == 0) {
        return;
    }

    for (int m = 0; m < M; ++m) {
        Y[m] = fmaf(alpha, myblas_sdot(N, &A[m * N], X), Y[m]);
    }
#endif
}

/// A = alpha*X*Y^t + A
/// - X: M vector
/// - Y: N vector
/// - A: M*N matrix
void myblas_sger(int M, int N, float alpha, const float *X, const float *Y, float *A) {
#if USE_BLAS
    cblas_sger(CblasRowMajor, M, N, alpha, X, 1, Y, 1, A, N);
#else
    if (alpha == 0) {
        return;
    }

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            A[m * N + n] = fmaf(alpha, X[m] * Y[n], A[m * N + n]);
        }
    }
#endif
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

typedef struct BMP_ColorPalette {
    uint32_t colors[256];
    size_t size;
} BMP_ColorPalette;

/// supports gray-scale only
/// **TODO**: full support
void BMP_init(BMP_Header *header, BMP_InfoHeader *info, BMP_ColorPalette *palette) {
    union {
        int8_t c[2];
        int16_t s;
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

    palette->size = 1u << info->bit_count;
    for (size_t i = 0; i < palette->size; ++i) {
        palette->colors[i] = (i << 16) | (i << 8) | i;
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
    const int size = rows * cols;
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

static FILE *mnist_fopen(const char *fname) {
    FILE *fp = fopen(fname, "rb");
    if (!fp) {
        fprintf(stderr, "%s: ", progname);
        perror(fname);
        exit(1);
    }
    return fp;
}

void mnist_init(
  float **train_x, int8_t *train_y, int *train_count, float **test_x, int8_t *test_y,
  int *test_count, int *rows, int *cols) {
    FILE *train_image = mnist_fopen(MNIST_TRAIN_IMAGES);
    FILE *train_label = mnist_fopen(MNIST_TRAIN_LABELS);
    FILE *test_image = mnist_fopen(MNIST_TEST_IMAGES);
    FILE *test_label = mnist_fopen(MNIST_TEST_LABELS);

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

void mnist_save_bmp(int32_t width, int32_t height, const float *x, const char *fname) {
    FILE *fp = fopen(fname, "wb");
    if (!fp) {
        fprintf(stderr, "%s: ", progname);
        perror(fname);
        exit(1);
    }

    uint8_t *data = malloc(sizeof(*data) * width * height);
    for (int64_t j = 0; j < width * height; ++j) {
        data[j] = x[j] * 256;
    }

    BMP_Header header;
    BMP_InfoHeader info;
    BMP_ColorPalette palette;
    BMP_init(&header, &info, &palette);

    info.width = width;
    info.height = height;

    fwrite(&header, 14, 1, fp);
    fwrite(&info, 40, 1, fp);
    fwrite(&palette.colors, sizeof(palette.colors[0]), palette.size, fp);
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
        fprintf(stderr, "%s: ", progname);
        perror(fname);
        exit(1);
    }

    BMP_Header header;
    BMP_InfoHeader info;
    BMP_ColorPalette palette;
    fread(&header, sizeof(header), 1, fp);
    fread(&info, sizeof(info), 1, fp);
    fread(palette.colors, sizeof(palette.colors[0]), 256, fp);
    union {
        int8_t c[2];
        int16_t s;
    } u = {.s = header.magic};
    if (
      u.c[0] != 'B' || u.c[1] != 'M' || info.header_size != 40 || info.bit_count != 8
      || info.width != MNIST_IMAGE_COLS || info.height != MNIST_IMAGE_ROWS) {
        error("%s: Unsupported file type", fname);
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
        fprintf(stderr, "%s: ", progname);
        perror(fname);
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
        fprintf(stderr, "%s: ", progname);
        perror(fname);
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
        error("%s: Unsupported file type", fname);
    }

    png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "%s: ", progname);
        perror(fname);
        exit(1);
    }

    info = png_create_info_struct(png);
    if (!png) {
        fprintf(stderr, "%s: ", progname);
        perror(fname);
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
        error("%s: Unsupported file type", fname);
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

// network {{{

/// y = W*x + b
/// - x: n vector
/// - W: m*n matrix
/// - b: m vector
static void affine(
  int m, int n, const float *restrict x, const float *restrict W, const float *restrict b,
  float *restrict y) {
    memcpy(y, b, m * sizeof(float));
    myblas_sgemv(m, n, 1, W, x, 1, y);
}

/// dW = dy * x^{T}
/// db = dy
/// dx = W^{T} * dy
/// - x:  n vector
/// - W:  m*n matrix
/// - dy: m vector
/// - dW: m*n matrix
/// - db: m vector
/// - dx: n vector
static void affine_bwd(
  int m, int n, const float *restrict x, const float *restrict W, const float *restrict dy,
  float *restrict dW, float *restrict db, float *restrict dx) {
    // dW = dy * x^{T}
    memset(dW, 0, m * n * sizeof(float));
    myblas_sger(m, n, 1, dy, x, dW);

    // db = dy
    memcpy(db, dy, m * sizeof(float));

    // dx = W^{T} * dy
#if USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasTrans, m, n, 1, W, n, dy, 1, 0, dx, 1);
#else
    memset(dx, 0, n * sizeof(float));
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) {
            dx[i] = fmaf(W[j * n + i], dy[j], dx[i]);
        }
    }
#endif
}

/// y_i = max(x_i, 0)
static void relu(int n, const float *restrict x, float *restrict y) {
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] > 0 ? x[i] : 0;
    }
}

static void relu_bwd(int n, const float *restrict x, const float *restrict dy, float *restrict dx) {
    for (int i = 0; i < n; ++i) {
        dx[i] = x[i] > 0 ? dy[i] : 0;
    }
}

/// y = exp(x .- max(x)) ./ sum(exp(x .- max(x)))
static void softmax(int n, const float *restrict x, float *restrict y) {
    int idx = myblas_ismax(n, x);
    double tmp = 0;
    for (int i = 0; i < n; ++i) {
        tmp += exp(x[i] - x[idx]);
    }
    for (int i = 0; i < n; ++i) {
        y[i] = exp(x[i] - x[idx]) / tmp;
    }
}

/// dx = y - t
/// - y: n vector
/// - t: n one-hot vector
static void softmax_with_loss_bwd(int n, const float *restrict y, int8_t t, float *restrict dx) {
    for (int i = 0; i < n; ++i) {
        dx[i] = y[i] - (t == i ? 1 : 0);
    }
}

static inline double cross_entropy(const float *y, int8_t t) {
    return -log(y[t] + 1e-7);
}

// }}} network

#define PARAMS_DIR DATA_DIR "params/"

#define LAYERS 3
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

enum Mode {
    kTrain,
    kInference,
    kSavePictures,
} mode;
int verbose;

typedef enum FileType {
    BMP,
#if USE_PNG
    PNG,
#endif
} FileType;

static struct FC_Param {
    float *W;
    float *b;
    float *dW[BATCH_SIZE];
    float *db[BATCH_SIZE];
    float *moment_W;
    float *moment_b;
} fc[LAYERS];
static const int fc_W_size[LAYERS] = {FC0_SIZE, FC1_SIZE, FC2_SIZE};
static const int fc_b_size[LAYERS] = {FC0_ROWS, FC1_ROWS, FC2_ROWS};
static struct NetworkDim {
    int in;
    int out;
} const dims[2 * LAYERS] = {{FC0_COLS, FC0_ROWS}, {FC0_ROWS, FC0_ROWS}, {FC1_COLS, FC1_ROWS},
                            {FC1_ROWS, FC1_ROWS}, {FC2_COLS, FC2_ROWS}, {FC2_ROWS, FC2_ROWS}};

static void initialize(void) {
    if (mode == kSavePictures) {
        return;
    }
    for (int i = 0; i < LAYERS; ++i) {
        fc[i].W = malloc(fc_W_size[i] * sizeof(float));
        fc[i].b = malloc(fc_b_size[i] * sizeof(float));
        if (mode == kTrain) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int j = 0; j < BATCH_SIZE; ++j) {
                fc[i].dW[j] = malloc(fc_W_size[i] * sizeof(float));
                fc[i].db[j] = malloc(fc_b_size[i] * sizeof(float));
            }
            fc[i].moment_W = calloc(fc_W_size[i], sizeof(float));
            fc[i].moment_b = calloc(fc_b_size[i], sizeof(float));
        }
    }
}

static void finalize(void) {
    if (mode == kSavePictures) {
        return;
    }
    for (int i = 0; i < LAYERS; ++i) {
        free(fc[i].W);
        free(fc[i].b);
        if (mode == kTrain) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int j = 0; j < BATCH_SIZE; ++j) {
                free(fc[i].dW[j]);
                free(fc[i].db[j]);
            }
            free(fc[i].moment_W);
            free(fc[i].moment_b);
        }
    }
}

static void load_param(float *dest, int len, const char *fname) {
    char str[32];
    snprintf(str, sizeof(str), PARAMS_DIR "%s", fname);
    FILE *fp = fopen(str, "rb");
    if (!fp) {
        for (int i = 0; i < len; ++i) {
            dest[i] = 0.05 * random_normal();
            // dest[i] = (random_uniform() - 0.5) * 0.2;
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
        fprintf(stderr, "%s: ", progname);
        perror(str);
        exit(1);
    }
    fwrite(src, sizeof(float), len, fp);
    fclose(fp);
}

static void random_shuffle(int n, float **x, int8_t *y) {
    for (int i = 0; i < n; ++i) {
        int idx = rand() / (RAND_MAX + 1.0) * n;
        SWAP(float *, x[i], x[idx]);
        SWAP(int8_t, y[i], y[idx]);
    }
}

static const char *__params_fname[LAYERS * 2]
  = {"fc0_W", "fc0_b", "fc1_W", "fc1_b", "fc2_W", "fc2_b"};

static void load_params(void) {
    for (int i = 0; i < LAYERS; ++i) {
        load_param(fc[i].W, fc_W_size[i], __params_fname[i * 2]);
        load_param(fc[i].b, fc_b_size[i], __params_fname[i * 2 + 1]);
    }
}

static void save_params(void) {
    for (int i = 0; i < LAYERS; ++i) {
        save_param(fc[i].W, fc_W_size[i], __params_fname[i * 2]);
        save_param(fc[i].b, fc_b_size[i], __params_fname[i * 2 + 1]);
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
        float *z[2 * LAYERS + 1];
        z[0] = x[i];
        for (int j = 0; j < 2 * LAYERS; ++j) {
            z[j + 1] = malloc(dims[j].out * sizeof(float));
        }

        affine(dims[0].out, dims[0].in, z[0], fc[0].W, fc[0].b, z[1]);
        relu(dims[1].in, z[1], z[2]);
        affine(dims[2].out, dims[2].in, z[2], fc[1].W, fc[1].b, z[3]);
        relu(dims[3].in, z[3], z[4]);
        affine(dims[4].out, dims[4].in, z[4], fc[2].W, fc[2].b, z[5]);
        softmax(dims[5].in, z[5], z[6]);

        float *y = z[2 * LAYERS];
        int idx = myblas_ismax(dims[2 * LAYERS - 1].out, y);
        c += idx == t[i];
        l += cross_entropy(y, t[i]);

        float *dx[2 * LAYERS];
        for (int j = 0; j < 2 * LAYERS; ++j) {
            dx[j] = malloc(dims[j].in * sizeof(float));
        }

        softmax_with_loss_bwd(dims[5].in, z[6], t[i], dx[5]);
        affine_bwd(dims[4].out, dims[4].in, z[4], fc[2].W, dx[5], fc[2].dW[i], fc[2].db[i], dx[4]);
        relu_bwd(dims[3].in, z[3], dx[4], dx[3]);
        affine_bwd(dims[2].out, dims[2].in, z[2], fc[1].W, dx[3], fc[1].dW[i], fc[1].db[i], dx[2]);
        relu_bwd(dims[1].in, z[1], dx[2], dx[1]);
        affine_bwd(dims[0].out, dims[0].in, z[0], fc[0].W, dx[1], fc[0].dW[i], fc[0].db[i], dx[0]);

        for (int j = 0; j < 2 * LAYERS; ++j) {
            free(z[j + 1]);
            free(dx[j]);
        }
    }

    float __eta = eta / BATCH_SIZE;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < LAYERS; ++i) {
        float *fc_dW_s = calloc(fc_W_size[i], sizeof(float));
        float *fc_db_s = calloc(fc_b_size[i], sizeof(float));

        for (int j = 0; j < BATCH_SIZE; ++j) {
            myblas_saxpy(fc_W_size[i], 1, fc[i].dW[j], fc_dW_s);
            myblas_saxpy(fc_b_size[i], 1, fc[i].db[j], fc_db_s);
        }

        myblas_sscal(fc_W_size[i], alpha, fc[i].moment_W);
        myblas_saxpy(fc_W_size[i], __eta, fc_dW_s, fc[i].moment_W);
        myblas_saxpy(fc_W_size[i], -1, fc[i].moment_W, fc[i].W);
        myblas_sscal(fc_b_size[i], alpha, fc[i].moment_b);
        myblas_saxpy(fc_b_size[i], __eta, fc_db_s, fc[i].moment_b);
        myblas_saxpy(fc_b_size[i], -1, fc[i].moment_b, fc[i].b);

        free(fc_dW_s);
        free(fc_db_s);
    }

    return (Result){-1, c, l};
}

static Result inference(float *x, int8_t t) {
    int c;
    double l;
    float *z[LAYERS * 2 + 1];
    z[0] = x;
    for (int i = 0; i < 2 * LAYERS; ++i) {
        z[i + 1] = malloc(dims[i].out * sizeof(float));
    }

    affine(dims[0].out, dims[0].in, z[0], fc[0].W, fc[0].b, z[1]);
    relu(dims[1].in, z[1], z[2]);
    affine(dims[2].out, dims[2].in, z[2], fc[1].W, fc[1].b, z[3]);
    relu(dims[3].in, z[3], z[4]);
    affine(dims[4].out, dims[4].in, z[4], fc[2].W, fc[2].b, z[5]);
    softmax(dims[5].in, z[5], z[6]);

    float *y = z[2 * LAYERS];
    int idx = myblas_ismax(dims[2 * LAYERS - 1].out, y);
    c = idx == t;
    l = cross_entropy(y, t);

    for (int i = 1; i < 2 * LAYERS + 1; ++i) {
        free(z[i]);
    }

    return (Result){idx, c, l};
}

static void train_main(int epochs, float eta, float decay, float alpha, int seed) {
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

    load_params();

    if (verbose) {
        printf("epoch,train_acc,train_loss,test_acc,test_loss\n");
    }
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
        if (verbose) {
            printf("%d,%.4f,%.4f", e + 1, c / (double)MNIST_TRAIN_COUNT, l / MNIST_TRAIN_COUNT);
        }

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
        if (verbose) {
            printf(",%.4f,%.4f\n", c / (double)MNIST_TEST_COUNT, l / MNIST_TEST_COUNT);
        }
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

static void display_image(int rows, int cols, const float *data) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float tmp = data[i * cols + j];
            if (tmp < 0.25) {
                putchar(' ');
            } else if (tmp < 0.5) {
                putchar('.');
            } else if (tmp < 0.75) {
                putchar('+');
            } else {
                putchar('*');
            }
            putchar(' ');
        }
        putchar('\n');
    }
}

static void inference_from_file(const char *fname) {
    FileType type;
    const char *ext = NULL;

    for (int i = strlen(fname); i >= 0; --i) {
        if (fname[i] == '.') {
            ext = &fname[i];
            break;
        }
    }

    if (!ext) {
        error("%s: Unsupported file type\n", fname);
    } else if (!strcmp(ext, ".bmp")) {
        type = BMP;
#if USE_PNG
    } else if (!strcmp(ext, ".png")) {
        type = PNG;
#endif
    } else {
        error("%s: Unsupported file type\n", fname);
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

    if (verbose) {
        display_image(MNIST_IMAGE_ROWS, MNIST_IMAGE_COLS, x);
    }

    printf("answer = %d\n", inference(x, 0).idx);
    free(x);
}

static void inference_main(int argc, char **argv) {
    load_params();
    for (int i = 0; i < argc; ++i) {
        inference_from_file(argv[i]);
    }
}

static int print_usage(void) {
    fprintf(
      stderr,
      "Usage: %s [OPTION]... [FILE]...\n"
      "Inference from each FILE when option `-i` is specified.\n"
      "Each FILE must be a 28*28 image file.\n"
      "Supported file types:\n"
      "  * Windows bitmap\n"
#if USE_PNG
      "  * PNG\n"
#endif
      "\n"
      "Modes:\n"
      "  -i         inference from FILE;\n"
      "  -p         convert datasets to pictures\n"
      "  -t         train (this is default)\n"
      "\n"
      "Parameters:\n"
      "  -d NUMBER  decay for learning rate (default: 1e-6)\n"
      "  -m NUMBER  momentum (default: 0.9)\n"
      "  -n NUMBER  number of epochs (default: 10)\n"
      "  -r NUMBER  learning rate (default: 0.01)\n"
      "  -s NUMBER  seed for random number (default: 1)\n"
      "\n"
      "  -v         verbose (default: on)\n"
      "  -q         suppress output\n"
      "  -h         display this help and exit\n",
      progname);
    return 1;
}

int main(int argc, char **argv) {
    static const char *opts = "d:ihm:n:pqr:s:tv";
    int epochs = 10;
    float eta = 0.01;
    float decay = 1e-6;
    float alpha = 0.9;
    unsigned seed = 1;
    mode = kTrain;
    verbose = 1;
    progname = argv[0];

    int c;
    while ((c = getopt(argc, argv, opts)) != -1) {
        switch (c) {
        case 'd':
            decay = strtof(optarg, NULL);
            break;
        case 'i':
            mode = kInference;
            break;
        case 'm':
            alpha = strtof(optarg, NULL);
            break;
        case 'n':
            epochs = atoi(optarg);
            break;
        case 'p':
            mode = kSavePictures;
            break;
        case 'q':
            verbose = 0;
            break;
        case 'r':
            eta = strtof(optarg, NULL);
            break;
        case 's':
            seed = strtoul(optarg, NULL, 10);
            break;
        case 't':
            mode = kTrain;
            break;
        case 'v':
            verbose = 1;
            break;
        default:
            return print_usage();
        }
    }

    initialize();

    switch (mode) {
    case kTrain:
        train_main(epochs, eta, decay, alpha, seed);
        break;
    case kInference:
        inference_main(argc - optind, argv + optind);
        break;
    case kSavePictures:
        mnist_save_images();
    }

    finalize();

    return 0;
}
