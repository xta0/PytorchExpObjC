//
//  PTHTensor.m
//  Pytorch-Exp-Demo
//
//  Created by taox on 8/25/19.
//  Copyright © 2019 taox. All rights reserved.
//
#import "PTHTensor.h"
#import "PTHTensor+Internal.h"
#import <PytorchExp/PytorchExp.h>

#define CHECK_IMPL(x) NSCAssert(x!=nil,@"impl is nil!");
#define CHECK_IMPL_(x) \
    CHECK_IMPL(x) \
    if (!x) { return nil; }

#define DEFINE_TENSOR_TYPES(_) \
    _(Byte) \
    _(Int) \
    _(Float) \
    _(Long) \
    _(Undefined)

#define DEFINE_TENSOR_SCALAR_TYPES(_) \
    _(Int) \
    _(Float) \
    _(Long) \

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

- (PTHTensorType)type {
    CHECK_IMPL(_impl)
    if(!_impl){
        return PTHTensorTypeUndefined;
    }
    return tensorType(_impl->scalar_type());
}

- (int64_t)capacity{
    CHECK_IMPL(_impl)
    if(!_impl){
        return -1;
    }
    return _impl->nbytes();
}

- (BOOL) quantized{
    CHECK_IMPL_(_impl)
    return _impl->is_quantized();
}

- (void* )data {
    CHECK_IMPL_(_impl);
    return _impl->unsafeGetTensorImpl()->storage().data();
}

- (int64_t) dim {
    CHECK_IMPL(_impl);
    if(!_impl) {
        return -1;
    }
    return _impl->dim();
}

+ (PTHTensor* )newWithType:(PTHTensorType)type Size:(NSArray<NSNumber* >*)size Data:(void* )data {
    return [self newWithType:type Size:size Data:data Quantized:NO];
}


+ (PTHTensor* )newWithType:(PTHTensorType)type Size:(NSArray<NSNumber* >*)size Data:(void* )data Quantized:(BOOL) quantized {
    if (!data || size.count == 0){
        return nil;
    }
    std::vector<int64_t> dimsVec;
    for (auto i = 0; i < size.count; ++i) {
        int64_t dim = size[i].integerValue;
        dimsVec.push_back(dim);
    }
    at::Tensor tensor = torch::from_blob( data, dimsVec,c10ScalarType(type));
    if (quantized) {
        tensor = at::quantize_linear(tensor, 1, 0, at::kQInt8);
    }
    return [PTHTensor newWithTensor:tensor];
}


- (NSString* )description {
    CHECK_IMPL_(_impl);
    return [NSString stringWithCString:_impl -> toString().c_str() encoding:NSASCIIStringEncoding];
}

- (PTHTensor* )objectAtIndexedSubscript:(NSUInteger)idx {
    CHECK_IMPL_(_impl)
    auto tensor = (*_impl)[idx];
    return [PTHTensor newWithTensor:tensor];
}

- (void)setObject:(id)obj atIndexedSubscript:(NSUInteger)idx {
    NSAssert(NO, @"Tensor object is immutable!");
}

#pragma mark NSCopying

- (instancetype)copyWithZone:(NSZone *)zone {
    return self;
}

@end

@implementation PTHTensor (Internal)

- (at::Tensor)toTensor {
    CHECK_IMPL(_impl);
    return at::Tensor(*_impl);
}

- (at::Tensor* )unsafeImpl {
    CHECK_IMPL_(_impl);
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
    t->_size = [shapes copy];
    t->_impl = std::move(impl);
    
    return t;
}


@end

@implementation PTHTensor (Operations)

#define DEFINE_SCALAR_OPS(op) \
    - (PTHTensor* )op##_:(float)value{ \
        CHECK_IMPL(_impl); \
        if(!_impl) { return nil; } \
        _impl->op##_(value); \
        return self; \
    }

DEFINE_SCALAR_OPS(add)
DEFINE_SCALAR_OPS(sub)
DEFINE_SCALAR_OPS(mul)
DEFINE_SCALAR_OPS(div)

- (NSNumber* )item {
    CHECK_IMPL_(_impl)
    switch (self.type) {
#define DEFINE_CASE(x) case PTHTensorType##x: return @(_impl->item().to##x());
            DEFINE_TENSOR_SCALAR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE
        default:
            return nil;
    }
}

- (PTHTensor* )to:(PTHTensorType) type {
    CHECK_IMPL_(_impl)
    c10::ScalarType scalarType = c10ScalarType(type);
    auto tensor = _impl->to(scalarType);
    return [PTHTensor newWithTensor:tensor];
}


- (PTHTensor* )permute:(NSArray<NSNumber* >*) dims {
    CHECK_IMPL_(_impl)
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
- (NSArray<PTHTensor* >* )topKResult:(NSNumber* )k
                                  Dim:(NSNumber* )dim
                            isLargest:(bool) largest
                             isSorted:(bool) sorted{
    CHECK_IMPL_(_impl)
    std::tuple<at::Tensor, at::Tensor> tensors = _impl->topk(k.unsignedIntegerValue, dim.integerValue, largest, sorted);
    auto idxs = std::get<1>(tensors);
    PTHTensor* firstEle  = [PTHTensor newWithTensor:std::get<0>(tensors)];
    PTHTensor* secondEle = [PTHTensor newWithTensor:std::get<1>(tensors)];
    return @[firstEle, secondEle];
}

- (PTHTensor* )view:(NSArray<NSNumber* >*)v {
    CHECK_IMPL_(_impl)
    std::vector<int64_t> views;
    for(NSNumber* n in v){
        views.push_back(n.unsignedIntegerValue);
    }
    auto newTensor = _impl->view(views);
    return [PTHTensor newWithTensor:newTensor];
}

@end
