#ifndef TIME_UTILS_XYW_HPP
#define TIME_UTILS_XYW_HPP
#include <chrono>
#include <string>
#include <time.h>

using namespace std;
/**
 * @brief Get the Local Time object
 * @ref https://www.runoob.com/w3cnote/cpp-time_t.html
 * @param y_m_d 
 * @param h_m_s 
 */
void getLocalTime(string y_m_d,string h_m_s){
    time_t timep;
    time(&timep);
    char ymd[16],hms[16];
    strftime(ymd,sizeof(ymd),"%Y-%m-%d",localtime(&timep));
    strftime(hms,sizeof(hms),"%H:%M:%S",localtime(&timep));
    y_m_d = ymd;
    h_m_s = hms;
}
/**
 * @brief 把UNIX时间戳拆成double，防止精度丢失
 * 
 * @param time 
 * @return double 
 */
inline double string2time(string time)
{
    string integer = time.substr(0, 10); // 整数
    string decimal = time.substr(10);    //小数
    double out = stoi(integer) + stod(decimal) / 1000000.0;
    return out;
}


/**
 * @brief 从字符串中拆出时间戳
 * 
 */
inline double getCameraTimestampfromFileName(string name)
{
    return string2time(name.substr(11, 16));
}
//toc的单位为ms
namespace tic
{
    // 计时
    class TicToc
    {
    private:
        std::chrono::time_point<std::chrono::system_clock> start, end;

    public:
        TicToc() { tic(); }

        void tic() { start = std::chrono::system_clock::now(); }

        double toc()
        {
            end = std::chrono::system_clock::now();

            std::chrono::duration<double> elapsed_seconds = end - start;
            return elapsed_seconds.count() * 1000;
        }
    };
    // 多阶段计时
    class TicTocPart
    {
    public:
        TicTocPart() { tic(); }

        void tic()
        {
            start = std::chrono::system_clock::now();
            tmp = start;
        }

        double toc()
        {
            std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = now - tmp;
            tmp = now;

            return elapsed_seconds.count() * 1000;
        }

        double tocEnd()
        {
            end = std::chrono::system_clock::now();

            std::chrono::duration<double> elapsed_seconds = end - start;
            return elapsed_seconds.count() * 1000;
        }

    private:
        std::chrono::time_point<std::chrono::system_clock> start, end, tmp;
    };
}
#endif