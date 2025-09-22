Lesson Examples
===============

This section shows common usage patterns from course lessons L01-L06.

Lesson 01: Neural Networks
--------------------------

Basic setup and training:

.. code-block:: python

   from introdl.utils import config_paths_keys, get_device
   from introdl.idlmam import train_simple_network

   # Setup environment
   paths = config_paths_keys()
   device = get_device()

   # Train neural network
   results = train_simple_network(model, loss_func, train_loader,
                                  device=device, epochs=600)

Lesson 02: CNNs
---------------

CNN training and visualization:

.. code-block:: python

   from introdl.idlmam import train_network
   from introdl.visul import create_image_grid, vis_feature_maps

   # Train CNN
   results = train_network(model, loss_func, train_loader,
                          test_loader=test_loader, device=device)

   # Visualize dataset and features
   create_image_grid(test_dataset, nrows=2, ncols=5, show_labels=True)
   vis_feature_maps(test_dataset, model, target_index=42)

Lesson 03: Training Techniques
-----------------------------

Advanced training with augmentation:

.. code-block:: python

   from introdl.utils import create_CIFAR10_loaders
   from introdl.visul import plot_training_metrics, plot_transformed_images

   # Create augmented data loaders
   train_loader, test_loader = create_CIFAR10_loaders(
       use_augmentation=True, batch_size=64
   )

   # Visualize augmentation effects
   plot_transformed_images(train_loader.dataset)

   # Train with advanced techniques
   results = train_network(model, loss_func, train_loader,
                          val_loader=val_loader, epochs=50)
   plot_training_metrics(results, [['train loss', 'val loss']])

Lessons 04-06: Advanced Topics
------------------------------

Architecture analysis and transfer learning:

.. code-block:: python

   # Model architecture analysis
   summarizer(model, input_size=(32, 3, 224, 224))

   # Transfer learning workflow
   pretrained_model = load_model(ResNet18, pretrained_checkpoint)
   results = train_network(pretrained_model, loss_func, train_loader)

   # Comprehensive evaluation
   evaluate_classifier(model, test_dataset, device)
