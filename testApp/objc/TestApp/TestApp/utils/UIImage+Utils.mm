//
//  Copyright © 2019年 Vizlab. All rights reserved.
//

#import "UIImage+Utils.h"

@implementation UIImage (Utils)

- (NSData* )rgb {
    
    CGImageRef inputCGImage = self.CGImage;
    NSUInteger width = CGImageGetWidth(inputCGImage);
    NSUInteger height = CGImageGetHeight(inputCGImage);
    NSUInteger bytesPerPixel = 4;
    NSUInteger bytesPerRow = bytesPerPixel * width;
    NSUInteger bitsPerComponent = 8;
    //decode image -> RGBA
    uint8_t * rawPixels = (uint8_t *) calloc(height * width * bytesPerPixel, sizeof(uint8_t));
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(rawPixels, width, height, bitsPerComponent, bytesPerRow, colorSpace, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), inputCGImage);
    CGColorSpaceRelease(colorSpace);
    CGContextRelease(context);
    //RGBA -> RGB
    uint8_t* rbgPixels = (uint8_t *) calloc(height * width * 3, sizeof(uint8_t));
    for(NSUInteger i=0; i<height * width; ++i){
        //skip the alpha channel
        rbgPixels[i*3+0] = rawPixels[i*4+0];
        rbgPixels[i*3+1] = rawPixels[i*4+1];
        rbgPixels[i*3+2] = rawPixels[i*4+2];
    }
    free(rawPixels);
    return [[NSData alloc]initWithBytes:rbgPixels length:height * width * 3 * sizeof(uint8_t)];;
}

- (UIImage* )resize:(CGSize)sz {
    
    if(CGSizeEqualToSize(self.size, sz)){
        return self;
    }
    UIGraphicsBeginImageContextWithOptions(sz, NO, 1);
    //decode the image to RGBA
    [self drawInRect:(CGRect){{0,0},{sz}}];
    UIImage* resizedImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return resizedImage;
}

- (UIImage* )rgbImage {
    
    uint8_t* buffer = (uint8_t* )[self rgb].bytes;
    size_t bitsPerComponent = 8;
    size_t bitsPerPixel = 24;
    size_t bytesPerRow = static_cast<size_t>(3 * self.size.width);
    CGColorSpaceRef colorSpaceRef = CGColorSpaceCreateDeviceRGB();
    CGBitmapInfo bitmapInfo = kCGImageAlphaNone;
    CGColorRenderingIntent renderingIntent = kCGRenderingIntentDefault;
    CGDataProviderRef provider = CGDataProviderCreateWithData(NULL, buffer, self.size.width*self.size.height*3  , NULL);
    CGImageRef imageRef = CGImageCreate(self.size.width,
                                        self.size.height,
                                        bitsPerComponent,
                                        bitsPerPixel,
                                        bytesPerRow,
                                        colorSpaceRef,
                                        bitmapInfo,
                                        provider,
                                        NULL, NO, renderingIntent);
    return [UIImage imageWithCGImage:imageRef];
}

@end
