//
//  ViewController.swift
//  Truck
//
//  Created by Amitesh Gupta on 20/07/24.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    

    @IBOutlet weak var result: UILabel!
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var camera: UIBarButtonItem!
    
    let imagePicker = UIImagePickerController()
    
    // CoreML model
    let imageClassifierWrapper: MobileNetV2?
    
    required init?(coder: NSCoder) {
        do {
            imageClassifierWrapper = try MobileNetV2(configuration: MLModelConfiguration())
        } catch {
            print("Failed to load CoreML model: \(error.localizedDescription)")
            imageClassifierWrapper = nil
        }
        super.init(coder: coder)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
                
                imagePicker.delegate = self
                imagePicker.sourceType = .photoLibrary
       
    }
    
        @IBAction func cameraButtonTapped(_ sender: UIBarButtonItem) {
            present(imagePicker, animated: true, completion: nil)
        }
        
        
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true, completion: nil)
        if let image = info[.originalImage] as? UIImage {
            imageView.image = image
            classifyImage(image: image)
        }
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }
    
   
    func classifyImage(image: UIImage) {
        guard let ciImage = CIImage(image: image) else {
            print("Unable to create CIImage from UIImage")
            result.text = "Error: Unable to create CIImage"
            return
        }
        
        guard let modelWrapper = imageClassifierWrapper else {
            print("Model is not loaded")
            result.text = "Error: Model is not loaded"
            return
        }
        
        guard let model = try? VNCoreMLModel(for: modelWrapper.model) else {
            print("Unable to load VNCoreMLModel")
            result.text = "Error: Unable to load VNCoreMLModel"
            return
        }
        
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            if let error = error {
                print("Failed to perform classification: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    self?.result.text = "Error: \(error.localizedDescription)"
                }
                return
            }
            
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else {
                print("Unexpected results or no results")
                DispatchQueue.main.async {
                    self?.result.text = "Error: Unexpected results"
                }
                return
            }
            
            DispatchQueue.main.async {
                self?.result.text = "Prediction: \(topResult.identifier) \nConfidence: \(topResult.confidence)"
              
            }
        }
        
        let handler = VNImageRequestHandler(ciImage: ciImage)
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                print("Failed to perform request: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    self.result.text = "Error: \(error.localizedDescription)"
                   
                }
            }
        }
    }
}
