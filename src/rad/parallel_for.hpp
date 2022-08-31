#pragma once

#include "opencv.hpp"

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

namespace rad
{
    template<typename T, typename Fun>
    void for_each(cv::Mat& mat, Fun&& fn)
    {
        cv::Size sz = mat.size();
        tbb::parallel_for(tbb::blocked_range2d<int>{0, sz.width, 0, sz.height},
                          [&mat, &fn](tbb::blocked_range2d<int> const& r) {
                              for (int w{r.rows().begin()}; w < r.rows().end(); ++w)
                              {
                                  for (int h{r.cols().begin()}; h < r.cols().end(); ++h)
                                  {
                                      cv::Point pt{w, h};
                                      T& pixel = mat.at<T>(pt);
                                      fn(pixel, pt);
                                  }
                              }
                          });
    }

    template<typename T, typename Fun>
    void for_each(cv::Mat const& mat, Fun&& fn)
    {
        cv::Size sz = mat.size();
        tbb::parallel_for(tbb::blocked_range2d<int>{0, sz.width, 0, sz.height},
                          [&mat, &fn](tbb::blocked_range2d<int> const& r) {
                              for (int w{r.rows().begin()}; w < r.rows().end(); ++w)
                              {
                                  for (int h{r.cols().begin()}; h < r.cols().end(); ++h)
                                  {
                                      cv::Point pt{w, h};
                                      T pixel = mat.at<T>(pt);
                                      fn(pixel, pt);
                                  }
                              }
                          });
    }

} // namespace rad
