//
//  PTHTensor.m
//  Pytorch-Exp-Demo
//
//  Created by taox on 8/25/19.
//  Copyright © 2019 taox. All rights reserved.
//

#include <string>
#import "PTHTensor.h"
#import <PytorchExp/PytorchExp.h>

#define CHECK_IMPL(x) NSCAssert(x!=nil,@"impl is nil!");

#define DEFINE_TENSOR_TYPES(_) \
    _(Byte) \
    _(Int) \
    _(Float) \
    _(Undefined)

static inline c10::ScalarType c10ScalarType(PTHTensorType type) {
    switch(type){
#define DEFINE_CASE(x) case PTHTensorType##x: return c10::ScalarType::x;
        DEFINE_TENSOR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE
    }
    return c10::ScalarType::Undefined;
}

static inline PTHTensorType tensorType(c10::ScalarType type) {
    switch(type){
#define DEFINE_CASE(x) case c10::ScalarType::x: return PTHTensorType##x;
        DEFINE_TENSOR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE
        default: return PTHTensorTypeUndefined;
    }
}

@implementation PTHTensor {
    std::shared_ptr<at::Tensor> _impl;
}


+ (PTHTensor* )newWithType:(PTHTensorType)type Shape:(NSArray<NSNumber* >*)dims Data:(void* )data {
    if (!data || dims.count == 0){
        return nil;
    }
    std::vector<int64_t> dimsVec;
    for (auto i = 0; i < dims.count; ++i) {
        int64_t dim = dims[i].integerValue;
        dimsVec.push_back(dim);
    }
    at::Tensor tensor = torch::from_blob( data, dimsVec,c10ScalarType(type));
    //pointer to at::tensor
    std::shared_ptr<at::Tensor> impl = std::make_shared<at::Tensor>(tensor);
    if(!impl){
        return nil;
    }
    
    PTHTensor* t = [PTHTensor new];
    t->_type  = type;
    t->_shape = [dims copy];
    t->_impl = std::move(impl);
    
    return t;
}

- (NSString* )description {
    CHECK_IMPL(_impl);
    return [NSString stringWithCString:_impl -> toString().c_str() encoding:NSASCIIStringEncoding];
}

@end

@implementation PTHTensor (Internal)

- (at::Tensor)toTensor {
    CHECK_IMPL(_impl);
    return at::Tensor(*_impl);
}

- (at::Tensor* )unsafeImpl {
    return _impl.get();
}

+ (PTHTensor* )newWithTensor:(const at::Tensor& ) tensor{
    std::shared_ptr<at::Tensor> impl = std::make_shared<at::Tensor>(tensor);
    if(!impl) {
        return nil;
    }
    PTHTensor* t = [PTHTensor new];
    NSMutableArray* shapes = [NSMutableArray new];
    auto dims = tensor.sizes();
    for (int i=0; i<dims.size(); ++i){
        [shapes addObject:@(dims[i])];
    }
    t->_shape = [shapes copy];
    t->_type = tensorType(tensor.scalar_type());
//    t->_data = impl->unsafeGetTensorImpl()->storage().data();
    t->_impl = std::move(impl);
    
    return t;
}

@end

@implementation PTHTensor (Operations)

- (PTHTensor* )toType:(PTHTensorType) type {
    c10::ScalarType scalarType = c10ScalarType(type);
    auto tensor = _impl->to(scalarType);
    return [PTHTensor newWithTensor:tensor];
}


- (PTHTensor* )permute:(NSArray<NSNumber* >*) dims {
    CHECK_IMPL(_impl);
    
    std::vector<int64_t> dimsVec;
    for (auto i = 0; i < dims.count; ++i) {
        int64_t dim = dims[i].integerValue;
        dimsVec.push_back(dim);
    }
    auto newTensor =  _impl->permute(dimsVec);
    newTensor.options();
    //tensors are immutable
    return [PTHTensor newWithTensor:newTensor];
}

@end
