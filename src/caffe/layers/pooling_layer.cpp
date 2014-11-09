// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filler.hpp"
//#include "caffe/PoolType.h"

#include <fstream>

using std::max;
using std::min;

namespace caffe {

 template<typename Dtype>
 void PoolingLayer<Dtype>::LoadPoolingStructure() {

   int n_pooled_elements, c, i, k;
   
   int top_size = pooled_width_*pooled_height_;//36;
  
       
    std::ifstream inFile(pooling_structure_file_.c_str());
   
    // shoud it be with the num? pooling_structure_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
    pooling_structure_.Reshape(channels_, pooled_height_*pooled_width_/*map_size*/, height_, width_); // I will use masks instead of indexing for efficiency
    
    //fill the structure with zeros
    FillerParameter filler_param;
	filler_param.set_value(0.);
    ConstantFiller<Dtype> filler(filler_param);
    filler.Fill(&(pooling_structure_));
    
	pooling_structure_.Update();
	
    Dtype* pooling_structure = pooling_structure_.mutable_cpu_data();
    
    int x_coordinate, y_coordinate;
    
    if (inFile.is_open()) {
    
      //load the mutable structure with ones where necessary
      for (c = 0; c < channels_; ++c) {
		for (i = 0; i < top_size; ++i) {
			
			//fill it with zeros
			/*for (int ph = 0; ph < height_; ++ph) {
				for (int pw = 0; pw < width_; ++pw) {
					pooling_structure[ph * width_ + pw] = 0;
				}
			}*/
			
			inFile >> n_pooled_elements;
		
			for (k = 0; k < n_pooled_elements; ++k) {
			  inFile >> x_coordinate;
			  inFile >> y_coordinate;
			  
			  //take care not to overstep the boundaries
			  x_coordinate = min(x_coordinate, width_- 1);
			  y_coordinate = min(y_coordinate, height_- 1);
			  
			  pooling_structure[y_coordinate*width_ + x_coordinate] = 1;
			
			}
	
			pooling_structure += pooling_structure_.offset(0,1); //move through map neurons
		}
		//pooling_structure += pooling_structure_.offset(1); //move through channels
      }
    }
    
    inFile.close();
 }

    


template <typename Dtype>
void PoolingLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "PoolingLayer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "PoolingLayer takes a single blob as output.";
  kernel_size_ = this->layer_param_.pooling_param().kernel_size();
  stride_ = this->layer_param_.pooling_param().stride();
  pad_ = this->layer_param_.pooling_param().pad();
  pooling_structure_file_= this->layer_param_.pooling_param().pooling_structure_file();
  
  if (pad_ != 0) {
    CHECK_EQ(this->layer_param_.pooling_param().pool(),
             PoolingParameter_PoolMethod_AVE)
        << "Padding implemented only for average pooling.";
  }
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_ - kernel_size_) / stride_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_ - kernel_size_) / stride_)) + 1;
  (*top)[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }
  // If selective max pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
    PoolingParameter_PoolMethod_MAX_SEL) {
            
    LoadPoolingStructure();
    
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
Dtype PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* pooling_structure;
    
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  int top_count = (*top)[0]->count();
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = -FLT_MAX;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_;
            int wstart = pw * stride_;
            int hend = min(hstart + kernel_size_, height_);
            int wend = min(wstart + kernel_size_, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] =
                  max(top_data[ph * pooled_width_ + pw],
                      bottom_data[h * width_ + w]);
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += (*top)[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_ - pad_;
            int wstart = pw * stride_ - pad_;
            int hend = min(hstart + kernel_size_, height_ + pad_);
            int wend = min(wstart + kernel_size_, width_ + pad_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += (*top)[0]->offset(0, 1);
      }
    }
    break;
    case PoolingParameter_PoolMethod_MAX_SEL:
    
      pooling_structure  = pooling_structure_.cpu_data();
      
    // Initialize
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = -FLT_MAX;
    }
    
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
	for (int nk = 0; nk < pooled_height_*pooled_width_; ++nk) //go through the neuron maps
	{
	  for (int h = 0; h < height_; ++h) {
	    for (int w = 0; w < width_; ++w) {
	      if(pooling_structure[h * width_ + w] == 1){
		//take this input into account
		  top_data[nk] =
			  max(top_data[nk],
			  bottom_data[h * width_ + w]);
	      }
	      
	    }
	  }
	  pooling_structure += pooling_structure_.offset(0,1); //move through map neurons
      }
      //pooling_structure += pooling_structure_.offset(1); //move through channels
      bottom_data += bottom[0]->offset(0, 1);
      top_data += (*top)[0]->offset(0, 1);
      }
    }
	
	//set those already not set to 0
	for (int i = 0; i < top_count; ++i) {
      if(top_data[i] == -FLT_MAX)
		top_data[i] = Dtype(0.);
    }
   break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  return Dtype(0.);
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* pooling_structure; // = this->pooling_structure_.mutable_cpu_data();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  memset(bottom_diff, 0, (*bottom)[0]->count() * sizeof(Dtype));
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_;
            int wstart = pw * stride_;
            int hend = min(hstart + kernel_size_, height_);
            int wend = min(wstart + kernel_size_, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                    top_diff[ph * pooled_width_ + pw] *
                    (bottom_data[h * width_ + w] ==
                        top_data[ph * pooled_width_ + pw]);
              }
            }
          }
        }
        // offset
        bottom_data += (*bottom)[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_ - pad_;
            int wstart = pw * stride_ - pad_;
            int hend = min(hstart + kernel_size_, height_ + pad_);
            int wend = min(wstart + kernel_size_, width_ + pad_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_data += (*bottom)[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
    case PoolingParameter_PoolMethod_MAX_SEL:
    
      pooling_structure = this->pooling_structure_.mutable_cpu_data();
    
    // The main loop
    for (int n = 0; n < (*bottom)[0]->num(); ++n) {
		pooling_structure = this->pooling_structure_.mutable_cpu_data();
		for (int c = 0; c < channels_; ++c) {
			for (int nk = 0; nk < pooled_height_*pooled_width_; ++nk) //go through the neuron maps
			{
			  for (int h = 0; h < height_; ++h) {
				for (int w = 0; w < width_; ++w) {
				  if(pooling_structure[h * width_ + w] == 1){
					//take this input into account
					bottom_diff[h * width_ + w] +=
							top_diff[nk] *
							(bottom_data[h * width_ + w] ==
								top_data[nk]);
				  }
				  
				}
			}
			pooling_structure += pooling_structure_.offset(0,1); //move through map neurons
		}
      
		//pooling_structure += pooling_structure_.offset(1); //move through channels
		//bottom_data += bottom[0]->offset(0, 1);
		//top_data += (*top)[0]->offset(0, 1);
		bottom_data += (*bottom)[0]->offset(0, 1);
		top_data += top[0]->offset(0, 1);
		bottom_diff += (*bottom)[0]->offset(0, 1);
		top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


INSTANTIATE_CLASS(PoolingLayer);


}  // namespace caffe
