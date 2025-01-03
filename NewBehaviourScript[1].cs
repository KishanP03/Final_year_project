using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NewBehaviourScript : MonoBehaviour
{
    private Transform headsetTransform;

    void Start()
    {
        Camera mainCamera = Camera.main;
        if (mainCamera != null)
        {
            headsetTransform = mainCamera.transform;
            Debug.Log("Headset Direction Logger Initialized.");
        }
        else
        {
            Debug.LogError("Main Camera not found. Ensure your XR Rig has a Main Camera.");
        }
    }

    void Update()
    {
        if (headsetTransform != null)
        {
            Vector3 headsetRotation = headsetTransform.rotation.eulerAngles;

            Vector3 headsetPosition = headsetTransform.position;

            Debug.Log("Headset Rotation (Euler Angles): " + headsetRotation + " " + "Headset Position: " + headsetPosition);
        }
        else
        {
            Debug.LogWarning("Headset transform is not assigned.");
        }
    }
}
