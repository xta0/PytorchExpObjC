//
//  TorchIValue.m
//  Pytorch-Exp-Demo
//
//  Created by taox on 8/25/19.
//  Copyright © 2019 taox. All rights reserved.
//

#include <vector>
#import "TorchIValue.h"
#import "TorchIValue+Internal.h"
#import "TorchTensor.h"
#import "TorchTensor+Internal.h"
#import <PytorchExp/PytorchExp.h>

#define IVALUE_TYPE(_) \
    _(Bool) \
    _(Int) \
    _(Double) \
    _(Tensor) \
    _(BoolList) \
    _(IntList) \
    _(DoubleList)\
    _(TensorList)


#define IVALUE_SCALAR_TYPE(_) \
    _(Bool, bool) \
    _(Int, int) \
    _(Double, double) \


#define DEFINE_IVALUE_WITH_SCALAR(type) \
+ (instancetype) newIValueWith##type:(NSNumber* )value{\
    if(![value isKindOfClass:[NSNumber class]]){ return nil; }\
    return [self newTorchIValueWithType:TorchIValueType##type Data:value]; \
}

#define DEFINE_IVALUE_WITH_SCALAR_LIST(type) \
+ (instancetype) newIValueWith##type##List:(NSArray<NSNumber*>* )list{\
if(![list isKindOfClass:[NSArray class]]){ return nil; }\
return [self newTorchIValueWithType:TorchIValueType##type##List Data:list]; \
}


@implementation TorchIValue {
    std::shared_ptr<at::IValue> _impl;
}

DEFINE_IVALUE_WITH_SCALAR(Bool)
DEFINE_IVALUE_WITH_SCALAR(Int)
DEFINE_IVALUE_WITH_SCALAR(Double)
DEFINE_IVALUE_WITH_SCALAR_LIST(Bool)
DEFINE_IVALUE_WITH_SCALAR_LIST(Int)
DEFINE_IVALUE_WITH_SCALAR_LIST(Double)

+ (instancetype) newIValueWithTensor:(TorchTensor* )tensor {
    if(![tensor isKindOfClass:[TorchTensor class]]){
        return nil;
    }
    auto t = tensor.toTensor;
    at::IValue atIValue(t);
    auto tmp = std::make_shared<at::IValue>(atIValue);
    if(!tmp) {
        return nil;
    }
    TorchIValue* value = [TorchIValue new];
    value->_type = TorchIValueTypeTensorList;
    value->_impl = std::move(tmp);
    return value;
}

+ (instancetype) newIValueWithTensorList:(NSArray<TorchTensor*>* )list {
    if(![list isKindOfClass:[NSArray<TorchTensor* > class]]){
        return nil;
    }
    c10::List<at::Tensor> tensorList;
    for(TorchTensor* tensor in list){
        auto t = tensor.toTensor;
        tensorList.push_back(t);
    }
    at::IValue atIValue(tensorList);
    auto tmp = std::make_shared<at::IValue>(atIValue);
    if(!tmp) {
        return nil;
    }
    TorchIValue* value = [TorchIValue new];
    value->_type = TorchIValueTypeTensorList;
    value->_impl = std::move(tmp);
    return value;
}

+ (instancetype) newTorchIValueWithType:(TorchIValueType)type Data:(id _Nullable)data {
    TorchIValue* value = [TorchIValue new];
    value->_type = type;
    at::IValue atIValue = {};
    switch (type) {
    #define  DEFINE_IVALUE_SCALAR_CASE(x,y) case TorchIValueType##x: {atIValue = at::IValue([(NSNumber* )data y##Value]);break;}
        IVALUE_SCALAR_TYPE(DEFINE_IVALUE_SCALAR_CASE)
    #undef DEFINE_IVALUE_SCALAR_CASE
            
    #define  DEFINE_IVALUE_SCALAR_LIST_CASE(x,y) case TorchIValueType##x##List: {\
    c10::List<y> list; \
    for(NSNumber* number in data){ list.push_back(number.y##Value); }\
    at::IValue value(list); break; }
        IVALUE_SCALAR_TYPE(DEFINE_IVALUE_SCALAR_LIST_CASE)
    #undef DEFINE_IVALUE_SCALAR_LIST_CASE
        default:
            break;
    }
    auto tmp = std::make_shared<at::IValue>(atIValue);
    value->_impl = std::move(tmp);
    return value->_impl ? value : nil;
}

#define DEFINE_TO_SCALAR(Type) \
- (NSNumber* )to##Type {\
if(!_impl || !_impl->is##Type()) { return nil; }\
return @(_impl->to##Type()); \
}

DEFINE_TO_SCALAR(Bool);
DEFINE_TO_SCALAR(Int);
DEFINE_TO_SCALAR(Double);

#define DEFINE_TO_SCALAR_LIST(Type) \
- (NSArray<NSNumber* >* )to##Type##List {\
if(!_impl || !_impl->is##Type()) { return nil; }\
auto list = _impl->to##Type##List(); \
NSMutableArray<NSNumber* >* tmp = [NSMutableArray new]; \
for(int i=0; i<list.size(); ++i) { [tmp addObject:@(list.get(i))]; } \
return [tmp copy];\
}

DEFINE_TO_SCALAR_LIST(Bool);
DEFINE_TO_SCALAR_LIST(Int);
DEFINE_TO_SCALAR_LIST(Double);
       

- (TorchTensor* )toTensor {
   if (!_impl || !_impl->isTensor()) {
       return nil;
   }
   at::Tensor tensor = _impl->toTensor();
   return [TorchTensor newWithTensor:tensor];
}

@end

#pragma mark - interoperability with at::IValue

@implementation TorchIValue (Internal)

- (at::IValue )toIValue {
    if(_impl){
        return at::IValue(*_impl);
    }
    return {};
}

- (at::IValue* )unsafeImpl {
    return _impl.get();
}

+ (TorchIValue* )newWithIValue:(const at::IValue& )v {
    TorchIValue* value = [TorchIValue new];
    
    #define IVALUE_TYPE_IF(x)\
        if(v.is##x()) { value->_type = TorchIValueType##x; }
        IVALUE_TYPE(IVALUE_TYPE_IF)
    #undef IVALUE_TYPE_IF

    auto tmp = std::make_shared<at::IValue>(v);
    if(!tmp){
        return nil;
    }
    
    value->_impl = std::move(tmp);
    return value;
}


@end

