#ifndef asd

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>
#include <numeric>
#include <tuple>
#include <fstream>
#include <cstdint>
#include <sstream>
#include <map>
#include <set>
#include <chrono>
#include <iomanip>
#include <cassert>

#pragma warning(disable : 4267)

typedef std::vector<double> avec;
typedef std::vector<int> ivec;
typedef std::vector<std::vector<double>> amat;
typedef std::vector<std::vector<int>> imat;

template<typename T> std::vector<T> operator + (std::vector<T> lhs, const std::vector<T>& rhs)
{
    if (lhs.size() != rhs.size()) throw std::length_error("vectors must be same size to add");
    for (size_t i = 0; i < lhs.size(); i++) lhs[i] = lhs[i] + rhs[i];
    return lhs;
}
template<typename T> std::vector<T> operator - (std::vector<T> lhs, const std::vector<T>& rhs)
{
    if (lhs.size() != rhs.size()) throw std::length_error("vectors must be same size to subtract");
    for (size_t i = 0; i < lhs.size(); i++) lhs[i] = lhs[i] - rhs[i];
    return lhs;
}
template<typename T> std::vector<T> operator * (std::vector<T> lhs, const std::vector<T>& rhs)
{
    if (lhs.size() != rhs.size()) throw std::length_error("vectors must be same size to multiply");
    for (size_t i = 0; i < lhs.size(); i++) lhs[i] *= rhs[i];
    return lhs;
}

template<typename T> std::vector<T> operator * (T lhs, std::vector<T> rhs)
{
    for (int i = 0; i < rhs.size(); i++)	rhs[i] = rhs[i] * lhs;

    return rhs;
}
template<typename T> std::vector<T> operator * (std::vector<T> rhs, T lhs)
{
    for (size_t i = 0; i < rhs.size(); i++) rhs[i] = rhs[i] * lhs;
    return rhs;
}
template<typename T> std::vector<T> operator / (std::vector<T> rhs, T lhs)
{
    for (size_t i = 0; i < rhs.size(); i++) rhs[i] = rhs[i] / lhs;
    return rhs;
}

template<typename T> std::vector<std::vector<T>> operator + (const std::vector<std::vector<T>>& lhs, const std::vector<std::vector<T>>& rhs)
{
    if (lhs.size() != rhs.size()) throw std::length_error("vectors must be same size");
    if (lhs[0].size() != rhs[0].size()) throw std::length_error("vectors must be same size");
    int rr = lhs.size(), cc = lhs[0].size();
    std::vector<std::vector<T>> result(rr, std::vector<T>(cc, 0));
    for (int r = 0; r < rr; r++)
        for (int c = 0; c < cc; c++)
            result[r][c] = lhs[r][c] + rhs[r][c];
    return result;
}
template<typename T> std::vector<std::vector<T>> operator - (const std::vector<std::vector<T>>& lhs, const std::vector<std::vector<T>>& rhs)
{
    if (lhs.size() != rhs.size()) throw std::length_error("vectors must be same size");
    if (lhs[0].size() != rhs[0].size()) throw std::length_error("vectors must be same size");
    int rr = lhs.size(), cc = lhs[0].size();
    std::vector<std::vector<T>> result(rr, std::vector<T>(cc, 0));
    for (int r = 0; r < rr; r++)
        for (int c = 0; c < cc; c++)
            result[r][c] = lhs[r][c] - rhs[r][c];
    return result;
}
template<typename T> std::vector<std::vector<T>> operator - (const std::vector<std::vector<T>> rhs, T lhs)
{
    int rr = rhs.size(), cc = rhs[0].size();
    std::vector<std::vector<T>> result(rr, std::vector<T>(cc, 0));
    for (int r = 0; r < rr; r++)
        for (int c = 0; c < cc; c++)
            result[r][c] = rhs[r][c] - lhs;
    return result;
}
template<typename T> std::vector<std::vector<T>> operator * (const std::vector<std::vector<T>> rhs, T lhs)
{
    int rr = rhs.size(), cc = rhs[0].size();
    std::vector<std::vector<T>> result(rr, std::vector<T>(cc, 0));
    for (int r = 0; r < rr; r++)
        for (int c = 0; c < cc; c++)
            result[r][c] = lhs * rhs[r][c];
    return result;
}
template<typename T> std::vector<std::vector<T>> operator / (const std::vector<std::vector<T>> rhs, T lhs)
{
    int rr = rhs.size(), cc = rhs[0].size();
    std::vector<std::vector<T>> result(rr, std::vector<T>(cc, 0));
    for (int r = 0; r < rr; r++)
        for (int c = 0; c < cc; c++)
            result[r][c] = rhs[r][c] / lhs;
    return result;
}


avec Relu(const avec& vec, double shift = 0.0)
{
    avec result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
        result[i] = (vec[i] > shift) ? vec[i] - shift : 0.0;
    return result;
}
avec Relu_derivative(const avec& vec, double shift = 0.0)
{
    avec result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
        result[i] = (vec[i] > shift) ? 1.0 : 0.0;
    return result;
}

avec softmax(const avec& x)
{
    double max_val = *std::max_element(x.begin(), x.end());
    avec out(x.size());
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); i++) { out[i] = std::exp(x[i] - max_val); sum += out[i]; }
    for (size_t i = 0; i < x.size(); i++) out[i] /= sum;
    return out;
}

double cross_entropy_loss(const amat& logits, const ivec& targets)
{
    double loss = 0.0;
    int T = logits.size();
    for (int t = 0; t < T; t++)
    {
        avec probs = softmax(logits[t]);
        loss -= std::log(probs[targets[t]] + 1e-9);
    }
    return loss / T;
}

amat cross_entropy_grad(const amat& logits, const ivec& targets)
{
    int T = logits.size();
    int V = logits[0].size();
    amat grad(T, avec(V, 0.0));
    for (int t = 0; t < T; t++)
    {
        avec probs = softmax(logits[t]);
        for (int v = 0; v < V; v++) grad[t][v] = probs[v];
        grad[t][targets[t]] -= 1.0;
        for (int v = 0; v < V; v++) grad[t][v] /= T;
    }
    return grad;
}

class TLinear
{
public:
    enum Function { linear, sigmoid, relu };
    int function = linear;

    amat w;              // [out_dim][in_dim]
    avec b;
    avec input;
    avec gradient;
    avec output;

    amat w_grad_accum;
    avec b_grad_accum;

    double bias = 1.0;

    TLinear() {}

    void init(int _function, int in_dim, int out_dim)
    {
        function = _function;
        std::mt19937 gen(42);
        std::normal_distribution<double> dist(0.0, 0.1);

        w.assign(out_dim, avec(in_dim, 0.0));
        for (int i = 0; i < out_dim; i++)
            for (int j = 0; j < in_dim; j++) w[i][j] = dist(gen);

        b.resize(out_dim);
        for (int i = 0; i < out_dim; i++) b[i] = dist(gen);

        output.resize(out_dim, 0.0);
        input.resize(in_dim, 0.0);
        gradient.resize(out_dim, 0.0);

        w_grad_accum.assign(out_dim, avec(in_dim, 0.0));
        b_grad_accum.assign(out_dim, 0.0);
    }

    avec forward(const avec& x)
    {
        input = x;
        int out_dim = w.size(), in_dim = x.size();
        output.assign(out_dim, 0.0);

        for (int i = 0; i < out_dim; i++)
        {
            for (int j = 0; j < in_dim; j++) output[i] += w[i][j] * x[j];
            output[i] += b[i] * bias;
        }

        if (function == relu) output = Relu(output);
        return output;
    }

    // Computes dx AND accumulates w/b gradients. Returns dx.
    avec backward(const avec& error)
    {
        gradient = error;
        if (function == relu) gradient = error * Relu_derivative(output);

        for (size_t out = 0; out < output.size(); ++out)
        {
            for (size_t inp = 0; inp < input.size(); ++inp)
                w_grad_accum[out][inp] += input[inp] * gradient[out];
            b_grad_accum[out] += bias * gradient[out];
        }

        avec dx(input.size(), 0.0);
        for (size_t inp = 0; inp < input.size(); inp++)
            for (size_t out = 0; out < output.size(); out++)
                dx[inp] += w[out][inp] * gradient[out];
        return dx;
    }

    void zero_grad()
    {
        int out_dim = w.size(), in_dim = out_dim > 0 ? w[0].size() : 0;
        w_grad_accum.assign(out_dim, avec(in_dim, 0.0));
        b_grad_accum.assign(out_dim, 0.0);
    }

    void update(double lr, int batch_size = 1)
    {
        double scale = lr / batch_size;
        for (size_t out = 0; out < output.size(); ++out)
        {
            for (size_t inp = 0; inp < input.size(); ++inp)
                w[out][inp] -= scale * w_grad_accum[out][inp];
            b[out] -= scale * b_grad_accum[out];
        }
    }
};
class Linear_Array
{
    TLinear ffn;
    amat cache_x;
    amat cache_y;

public:
    Linear_Array() {}

    void init(int function, int in_dim, int out_dim)
    {
        ffn.init(function, in_dim, out_dim);
    }

    amat forward(const amat& x)
    {
        cache_x = x;
        int T = x.size();
        amat y(T, avec(ffn.w.size(), 0.0));
        for (int t = 0; t < T; t++) y[t] = ffn.forward(x[t]);
        cache_y = y;
        return y;
    }

    // Restores ffn state per token, calls TLinear::backward (which accumulates).
    // Returns dX across all token positions.
    amat backward(const amat& dY)
    {
        int T = dY.size();
        amat dX(T, avec(cache_x[0].size(), 0.0));
        for (int t = 0; t < T; t++)
        {
            ffn.output = cache_y[t];
            ffn.input = cache_x[t];
            dX[t] = ffn.backward(dY[t]);   // accumulates into ffn's accumulators
        }
        return dX;
    }

    void zero_grad() { ffn.zero_grad(); }

    void update(double lr, int batch_size = 1) { ffn.update(lr, batch_size); }
};
class Linear_Array_3
{
    Linear_Array ffn1, ffn2, ffn3;

public:
    Linear_Array_3() {}

    void init(int dim, int /*max_len*/)
    {
        ffn1.init(TLinear::linear, dim, dim * 4);
        ffn2.init(TLinear::relu, dim * 4, dim * 4);
        ffn3.init(TLinear::linear, dim * 4, dim);
    }

    amat forward(amat x)
    {
        x = ffn1.forward(x);
        x = ffn2.forward(x);
        x = ffn3.forward(x);
        return x;
    }

    amat backward(amat error)
    {
        error = ffn3.backward(error);
        error = ffn2.backward(error);
        error = ffn1.backward(error);
        return error;
    }

    void zero_grad() { ffn1.zero_grad(); ffn2.zero_grad(); ffn3.zero_grad(); }

    void update(double lr, int batch_size = 1)
    {
        ffn1.update(lr, batch_size);
        ffn2.update(lr, batch_size);
        ffn3.update(lr, batch_size);
    }
};

class LayerNorm
{
public:
    avec gamma, beta;
    avec last_x;
    avec gamma_grad_accum;
    avec beta_grad_accum;

    LayerNorm() {}

    void init(int dim)
    {
        gamma.assign(dim, 1.0);
        beta.assign(dim, 0.0);
        gamma_grad_accum.assign(dim, 0.0);
        beta_grad_accum.assign(dim, 0.0);
    }

    avec forward(avec x)
    {
        last_x = x;
        int d = x.size();
        double mean = std::accumulate(x.begin(), x.end(), 0.0) / d;
        double var = 0.0;
        for (int i = 0; i < d; i++) var += (x[i] - mean) * (x[i] - mean);
        double std_inv = 1.0 / std::sqrt(var / d + 1e-5);
        avec out(d);
        for (int i = 0; i < d; i++) out[i] = gamma[i] * (x[i] - mean) * std_inv + beta[i];
        return out;
    }

    // Accumulates gamma/beta gradients. Returns dx. Never touches weights.
    avec backward(const avec& d_out)
    {
        int d = last_x.size();
        double mean = std::accumulate(last_x.begin(), last_x.end(), 0.0) / d;
        double var = 0.0;
        for (int i = 0; i < d; i++) var += (last_x[i] - mean) * (last_x[i] - mean);
        double std_inv = 1.0 / std::sqrt(var / d + 1e-5);

        avec x_hat(d);
        for (int i = 0; i < d; i++) x_hat[i] = (last_x[i] - mean) * std_inv;

        for (int i = 0; i < d; i++)
        {
            gamma_grad_accum[i] += d_out[i] * x_hat[i];
            beta_grad_accum[i] += d_out[i];
        }

        avec d_x_hat(d);
        for (int i = 0; i < d; i++) d_x_hat[i] = d_out[i] * gamma[i];

        double sum1 = 0.0, sum2 = 0.0;
        for (int i = 0; i < d; i++) { sum1 += d_x_hat[i]; sum2 += d_x_hat[i] * x_hat[i]; }

        avec dx(d);
        for (int i = 0; i < d; i++)
            dx[i] = std_inv * (d_x_hat[i] - (sum1 + x_hat[i] * sum2) / d);
        return dx;
    }

    amat forward(amat x)
    {
        for (size_t i = 0; i < x.size(); i++) x[i] = forward(x[i]);
        return x;
    }

    // Accumulates across all tokens, returns dX. Never touches weights.
    amat backward(const amat& d_out)
    {
        amat dx(d_out.size(), avec(d_out[0].size(), 0.0));
        for (size_t i = 0; i < d_out.size(); i++) dx[i] = backward(d_out[i]);
        return dx;
    }

    void zero_grad()
    {
        gamma_grad_accum.assign(gamma.size(), 0.0);
        beta_grad_accum.assign(beta.size(), 0.0);
    }

    void update(double lr, int batch_size = 1)
    {
        double scale = lr / batch_size;
        for (size_t i = 0; i < gamma.size(); i++)
        {
            gamma[i] -= scale * gamma_grad_accum[i];
            beta[i] -= scale * beta_grad_accum[i];
        }
    }
};

class TAttentionALiBi
{
public:
    int embed_size;
    double slope;
    amat last_input;
    amat last_attn;
    int rotary = 1;

    TAttentionALiBi() : embed_size(0), slope(1.0) {}
    void init(int _embed_size, double _slope = 1.0) { embed_size = _embed_size; slope = _slope; }
    amat forward(const amat& x)
    {
        last_input = x;

        int T = x.size(), D = x[0].size();
        amat output(T, avec(D, 0.0));
        last_attn.assign(T, avec(T, 0.0));
        double scale = 1.0 / std::sqrt((double)D);

        for (int i = 0; i < T; ++i)
        {
            avec scores(T, -1e9);
            for (int j = 0; j <= i; ++j)
            {
                double dot = 0.0;
                for (int d = 0; d < D; ++d) dot += x[i][d] * x[j][d];
                scores[j] = dot * scale - slope * (i - j); // (i - j) is relative position embeding
                //scores[j] = dot * scale;
            }

            double max_s = *std::max_element(scores.begin(), scores.end());
            double sum_e = 0.0;
            for (int j = 0; j <= i; ++j) { scores[j] = std::exp(scores[j] - max_s); sum_e += scores[j]; }
            for (int j = 0; j <= i; ++j) { scores[j] /= sum_e; last_attn[i][j] = scores[j]; }
            for (int j = 0; j <= i; ++j)
                for (int d = 0; d < D; ++d) output[i][d] += scores[j] * x[j][d];
        }
 
        return output;
    }
    amat backward(const amat& dout)
    {
        int T = last_input.size(), D = last_input[0].size();
        amat dx(T, avec(D, 0.0));
        double scale = 1.0 / std::sqrt((double)D);

        for (int i = 0; i < T; ++i)
        {
            // ----- Step 1: dL/d(attn) -----
            avec dattn(T, 0.0);
            for (int j = 0; j <= i; ++j)
                for (int d = 0; d < D; ++d) dattn[j] += dout[i][d] * last_input[j][d];

            // ----- Step 2: softmax backward -----        // ds = softmax_grad
            avec ds(T, 0.0);
            double dot = 0.0;
            for (int j = 0; j <= i; ++j) dot += dattn[j] * last_attn[i][j];
            for (int j = 0; j <= i; ++j) ds[j] = last_attn[i][j] * (dattn[j] - dot);

            // ----- Step 3: score backward -----
            for (int j = 0; j <= i; ++j)
                for (int d = 0; d < D; ++d)
                {
                    dx[i][d] += ds[j] * last_input[j][d] * scale;
                    dx[j][d] += ds[j] * last_input[i][d] * scale;
                }

            // ----- Step 4: value path -----
            for (int j = 0; j <= i; ++j)
                for (int d = 0; d < D; ++d) dx[j][d] += last_attn[i][j] * dout[i][d];
        }
        return dx;
    }

    void zero_grad() {}
    void update(double lr, int batch_size = 1) {}
};

class Embeding
{
public:
    int dim = 0;
    amat vocabulary;
    ivec tokens;
    amat vocab_grad_accum;

    void init(int vocabulary_size, int embed_dim, int initialization = 1)
    {
        dim = embed_dim;
        vocabulary.clear();
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-0.1, 0.1);

        for (int tok = 0; tok < vocabulary_size; tok++)
        {
            avec vecs(embed_dim, 0.0);
            if (initialization == 1)
                for (int i = 0; i < embed_dim; i++) vecs[i] = dist(rng);
            vocabulary.push_back(vecs);
        }
        vocab_grad_accum.assign(vocabulary_size, avec(embed_dim, 0.0));
    }

    amat forward(const ivec& _tokens)
    {
        tokens = _tokens;
        amat embeddings;
        for (int token_id : tokens) embeddings.push_back(vocabulary[token_id]);
        return embeddings;
    }

    void backward(const amat& errors)
    {
        for (size_t i = 0; i < errors.size(); i++)
        {
            int token_id = tokens[i];
            for (int j = 0; j < dim; j++)
                vocab_grad_accum[token_id][j] += errors[i][j];
        }
    }

    void zero_grad()
    {
        vocab_grad_accum.assign(vocabulary.size(), avec(dim, 0.0));
    }

    void update(double lr, int batch_size = 1)
    {
        double scale = lr / batch_size;
        for (size_t tok = 0; tok < vocabulary.size(); tok++)
            for (int j = 0; j < dim; j++)
            {
                vocabulary[tok][j] -= scale * vocab_grad_accum[tok][j];
                vocab_grad_accum[tok][j] = 0.0;
            }
    }
};

struct Config
{
    int vocabulary_size = 27;
    int dim = 32;
    int layers = 2;
    int seq_length = 16; // 16; // block_size =  how long each context sequence is, also known as T;
    double lr = 0.01;
    int epochs = 1000;
    int batch_size = 8;
} con;

class Data
{
public:
    std::string txt;
    std::vector<char> chars;
    std::map<char, int> char2idx;
    std::map<int, char> idx2char;
    ivec corpus;

    void prepare(const std::string& input_text)
    {
        std::ifstream f(input_text);
        if (f.good())
        {
            txt = std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            std::cout << "Loaded file: " << input_text << " (" << txt.size() << " chars)\n";
        }
        else
        {
            std::cout << "No file found. Using built-in training text.\n";
            txt = "hello world this is a simple transformer test the model learns character level language modeling ";
            std::string base = txt;
            for (int i = 0; i < 10; i++) txt += base;
        }

        std::set<char> charset(txt.begin(), txt.end());
        chars = std::vector<char>(charset.begin(), charset.end());
        con.vocabulary_size = chars.size();

        for (size_t i = 0; i < chars.size(); i++) { char2idx[chars[i]] = i; idx2char[i] = chars[i]; }

        corpus.clear();
        for (char c : txt) corpus.push_back(char2idx[c]);

        std::cout << "Vocab size: " << con.vocabulary_size << "\n";
        std::cout << "Corpus length: " << corpus.size() << "\n";
    }

    std::tuple<ivec, ivec> getBatch(std::mt19937& rng)
    {
        int max_start = corpus.size() - con.seq_length - 1;
        std::uniform_int_distribution<int> dist(0, max_start);
        int start = dist(rng);
        ivec input(corpus.begin() + start, corpus.begin() + start + con.seq_length);
        ivec target(corpus.begin() + start + 1, corpus.begin() + start + con.seq_length + 1);
        return { input, target };
    }
};

class TBlock
{
    int dim = -1;

    TAttentionALiBi att; // + relative position'
    //TAttentionRoPE att;
    //AttWeightedSA att; // no pos
    //AttWeightedSM att; // no pos

    Linear_Array_3  ffn;
    LayerNorm       norm1, norm2;

public:
    TBlock() {}
    TBlock(int _dim) : dim(_dim)
    {
        att.init(dim);
        ffn.init(dim, con.seq_length);
        norm1.init(dim);
        norm2.init(dim);
    }

    amat forward(amat x)
    {
        x = x + att.forward(norm1.forward(x));
        x = x + ffn.forward(norm2.forward(x));
        return x;
    }
    amat backward(amat error)
    {
        error = error + norm2.backward(ffn.backward(error));
        error = error + norm1.backward(att.backward(error));
        return error;
    }

    void zero_grad() { ffn.zero_grad(); norm1.zero_grad(); norm2.zero_grad(); att.zero_grad(); }
    void update(double lr, int batch_size = 1)
    {
        ffn.update(lr, batch_size);
        norm1.update(lr, batch_size);
        norm2.update(lr, batch_size);
        att.update(lr, batch_size);
    }
};

class Transformer
{
public:
    Embeding            embeding;
    std::vector<TBlock> layers;
    Linear_Array        head;
    LayerNorm           final_norm;
    amat                logits;

    void init()
    {
        embeding.init(con.vocabulary_size, con.dim, 1);
        layers.clear();
        for (int i = 0; i < con.layers; i++) layers.emplace_back(con.dim);
        head.init(TLinear::linear, con.dim, con.vocabulary_size);
        final_norm.init(con.dim);
    }

    amat forward(const ivec& tokens)
    {
        amat x = embeding.forward(tokens);
        for (auto& layer : layers) x = layer.forward(x);
        x = final_norm.forward(x);
        logits = head.forward(x);
        return logits;
    }

    void backward(const amat& d_logits)
    {
        amat dx = head.backward(d_logits);
        dx = final_norm.backward(dx);
        for (int i = (int)layers.size() - 1; i >= 0; i--) dx = layers[i].backward(dx);
        embeding.backward(dx);
    }

    void zero_grad()
    {
        head.zero_grad();
        final_norm.zero_grad();
        for (auto& layer : layers) layer.zero_grad();
        embeding.zero_grad();
    }

    void update(double lr, int batch_size = 1)
    {
        head.update(lr, batch_size);
        final_norm.update(lr, batch_size);
        for (auto& layer : layers) layer.update(lr, batch_size);
        embeding.update(lr, batch_size);
    }

    int predict_next(const ivec& context)
    {
        amat lg = forward(context);
        //avec& last_logits = lg.back();
        //avec& last_logits = lg.back(); // 000 original - do not delete this comment!
        auto last_logits = lg.back();

        int best = 0;
        for (size_t i = 1; i < last_logits.size(); i++)  if (last_logits[i] > last_logits[best]) best = i;
        return best;
    }
};

Data        data;
Transformer trans;

void train()
{
    std::cout << "\n=== Training (batch_size=" << con.batch_size << ") ===\n";

    // So steps here really means "how many random batches we draw per epoch", not a strict full pass over the data.
    int steps = 20; 
    std::cout << "steps: " << steps << std::endl;

    std::mt19937 rng(123);

    auto total_start = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < con.epochs; epoch++)
    {
        auto epoch_start = std::chrono::steady_clock::now();

        double total_loss = 0.0;

        //for (int start = 0; start + con.seq_length + 1 < (int)data.corpus.size(); start += con.seq_length)
        for (int step = 0; step < steps; step++)
        {
            trans.zero_grad();
            double batch_loss = 0.0;

            for (int b = 0; b < con.batch_size; b++)
            {
                ivec input_tokens, target_tokens;
                std::tie(input_tokens, target_tokens) = data.getBatch(rng);

                amat logits = trans.forward(input_tokens);
                batch_loss += cross_entropy_loss(logits, target_tokens);
                trans.backward(cross_entropy_grad(logits, target_tokens));
            }

            trans.update(con.lr, con.batch_size);
            total_loss += batch_loss / con.batch_size;
        }

        auto epoch_end = std::chrono::steady_clock::now();
        double epoch_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();

        auto elapsed = std::chrono::duration<double>(epoch_end - total_start).count();
        double eta_sec = (epoch + 1 < con.epochs)
            ? epoch_sec * (con.epochs - epoch - 1)
            : 0.0;

        if ((epoch + 1) % 10 == 0 || epoch == 0)
        {
            std::cout << "Epoch " << std::setw(4) << epoch + 1 << "/" << con.epochs
                << "  avg_loss=" << std::fixed << std::setprecision(4) << total_loss / steps
                << "  epoch=" << std::setprecision(2) << epoch_sec << "s"
                << "  elapsed=" << elapsed << "s"
                << "  ETA=" << eta_sec << "s"
                << "\n";
        }
    }

    auto total_end = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - total_start).count();
    std::cout << "Training complete.  Total time: " << std::fixed << std::setprecision(2) << total_sec << "s\n";
}

void test()
{
    std::cout << "\n=== Test (text generation) ===\n";
    if ((int)data.corpus.size() < con.seq_length + 1) { std::cout << "Not enough data.\n"; return; }

    ivec prompt(data.corpus.begin(), data.corpus.begin() + con.seq_length);

    std::cout << "Prompt: \"";
    for (int tok : prompt) std::cout << data.idx2char[tok];

    std::cout << "\"\nGenerated: \"";
    for (int tok : prompt) std::cout << data.idx2char[tok];

    ivec context = prompt;
    for (int i = 0; i < 60; i++)
    {
        int next = trans.predict_next(context);
        std::cout << data.idx2char[next];
        std::cout.flush();
        context.erase(context.begin());
        context.push_back(next);
    }
    std::cout << "\"\n";

    std::mt19937 rng(999);
    double test_loss = 0.0;
    ivec inp, tgt;
    for (int i = 0; i < 10; i++)
    {
        std::tie(inp, tgt) = data.getBatch(rng);
        test_loss += cross_entropy_loss(trans.forward(inp), tgt);
    }
    std::cout << "Test avg loss over 10 sequences: " << test_loss / 10 << "\n";
}

int main()
{
    std::cout << "Simple Transformer (C++) - Batched Training\n";
    std::cout << "dim=" << con.dim << "\n layers=" << con.layers
        << "\n seq=" << con.seq_length << "\n lr=" << con.lr
        << "\n epochs=" << con.epochs << "\n batch_size=" << con.batch_size << "\n";

    data.prepare("input.txt");
    //data.prepare("alica.txt");

    trans.init();
    train();
    test();

    system("pause");
    return 0;
}

#endif