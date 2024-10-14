import json
import random
import chainlit as cl
from datetime import datetime, timedelta

# Function Definitions
check_order_status_def = {
    "name": "check_order_status",
    "description": "Check the status of a customer's order",
    "parameters": {
      "type": "object",
      "properties": {
        "customer_id": {
          "type": "string",
          "description": "The unique identifier for the customer"
        },
        "order_id": {
          "type": "string",
          "description": "The unique identifier for the order"
        }
      },
      "required": ["customer_id", "order_id"]
    }
}

process_return_def = {
    "name": "process_return",
    "description": "Initiate a return process for a customer's order",
    "parameters": {
      "type": "object",
      "properties": {
        "customer_id": {
          "type": "string",
          "description": "The unique identifier for the customer"
        },
        "order_id": {
          "type": "string",
          "description": "The unique identifier for the order to be returned"
        },
        "reason": {
          "type": "string",
          "description": "The reason for the return"
        }
      },
      "required": ["customer_id", "order_id", "reason"]
    }
}

get_product_info_def = {
    "name": "get_product_info",
    "description": "Retrieve information about a specific product",
    "parameters": {
      "type": "object",
      "properties": {
        "customer_id": {
          "type": "string",
          "description": "The unique identifier for the customer"
        },
        "product_id": {
          "type": "string",
          "description": "The unique identifier for the product"
        }
      },
      "required": ["customer_id", "product_id"]
    }
}

update_account_info_def = {
    "name": "update_account_info",
    "description": "Update a customer's account information",
    "parameters": {
      "type": "object",
      "properties": {
        "customer_id": {
          "type": "string",
          "description": "The unique identifier for the customer"
        },
        "field": {
          "type": "string",
          "description": "The account field to be updated (e.g., 'email', 'phone', 'address')"
        },
        "value": {
          "type": "string",
          "description": "The new value for the specified field"
        }
      },
      "required": ["customer_id", "field", "value"]
    }
}


  
cancel_order_def = {  
    "name": "cancel_order",  
    "description": "Cancel a customer's order before it is processed",  
    "parameters": {  
        "type": "object",  
        "properties": {  
            "customer_id": {
                "type": "string",
                "description": "The unique identifier for the customer"
            },
            "order_id": {  
                "type": "string",  
                "description": "The unique identifier of the order to be cancelled"  
            },  
            "reason": {  
                "type": "string",  
                "description": "The reason for cancelling the order"  
            }  
        },  
        "required": ["customer_id", "order_id", "reason"]  
    }  
}  

schedule_callback_def = {  
    "name": "schedule_callback",  
    "description": "Schedule a callback with a customer service representative",  
    "parameters": {  
        "type": "object",  
        "properties": {  
            "customer_id": {
                "type": "string",
                "description": "The unique identifier for the customer"
            },
            "callback_time": {  
                "type": "string",  
                "description": "Preferred time for the callback in ISO 8601 format"  
            }  
        },  
        "required": ["customer_id", "callback_time"]  
    }  
}  

get_customer_info_def = {  
    "name": "get_customer_info",  
    "description": "Retrieve information about a specific customer",  
    "parameters": {  
        "type": "object",  
        "properties": {  
            "customer_id": {  
                "type": "string",  
                "description": "The unique identifier for the customer"  
            }  
        },  
        "required": ["customer_id"]  
    }  
}  
  

  
async def cancel_order_handler(customer_id, order_id, reason):  
    status = "Cancelled"
    
    # Generate random cancellation details
    cancellation_date = datetime.now()
    refund_amount = round(random.uniform(10, 500), 2)
    
    # Read the HTML template
    with open('order_cancellation_template.html', 'r') as file:
        html_content = file.read()
    
    # Replace placeholders with actual data
    html_content = html_content.format(
        order_id=order_id,
        customer_id=customer_id,
        cancellation_date=cancellation_date.strftime("%B %d, %Y"),
        refund_amount=refund_amount,
        status=status
    )
    
    # Return the Chainlit message with HTML content
    await cl.Message(content=f"Your order has been cancelled. Here are the details:\n{html_content}").send()
    return f"Order {order_id} for customer {customer_id} has been cancelled. Reason: {reason}. A confirmation email has been sent."  
  
async def schedule_callback_handler(customer_id, callback_time):  
        # Read the HTML template
    with open('callback_schedule_template.html', 'r') as file:
        html_content = file.read()

    # Replace placeholders with actual data
    html_content = html_content.format(
        customer_id=customer_id,
        callback_time=callback_time
    )

    # Return the Chainlit message with HTML content
    await cl.Message(content=f"Your callback has been scheduled. Here are the details:\n{html_content}").send()
    return f"Callback scheduled for customer {customer_id} at {callback_time}. A representative will contact you then."
  
async def check_order_status_handler(customer_id, order_id):
    status = "In Transit"
    
    # Generate random order details
    order_date = datetime.now() - timedelta(days=random.randint(1, 10))
    estimated_delivery = order_date + timedelta(days=random.randint(3, 7))
    # Read the HTML template
    with open('order_status_template.html', 'r') as file:
        html_content = file.read()

    # Replace placeholders with actual data
    html_content = html_content.format(
        order_id=order_id,
        customer_id=customer_id,
        order_date=order_date.strftime("%B %d, %Y"),
        estimated_delivery=estimated_delivery.strftime("%B %d, %Y"),
        status=status
    )

    # Return the Chainlit message with HTML content
    await cl.Message(content=f"Here is the detail of your order \n {html_content}").send()
    return f"Order {order_id} status for customer {customer_id}: {status}"
    

async def process_return_handler(customer_id, order_id, reason):
    return f"Return for order {order_id} initiated by customer {customer_id}. Reason: {reason}. Please expect a refund within 5-7 business days."

async def get_product_info_handler(customer_id, product_id):
    products = {
        "P001": {"name": "Wireless Earbuds", "price": 79.99, "stock": 50},
        "P002": {"name": "Smart Watch", "price": 199.99, "stock": 30},
        "P003": {"name": "Laptop Backpack", "price": 49.99, "stock": 100}
    }
    product_info = products.get(product_id, "Product not found")
    return f"Product information for customer {customer_id}: {json.dumps(product_info)}"

async def update_account_info_handler(customer_id, field, value):
    return f"Account information updated for customer {customer_id}. {field.capitalize()} changed to: {value}"

async def get_customer_info_handler(customer_id):  
    # Simulated customer data (using placeholder information)  
    customers = {  
        "C001": {"membership_level": "Gold", "account_status": "Active"},  
        "C002": {"membership_level": "Silver", "account_status": "Pending"},  
        "C003": {"membership_level": "Bronze", "account_status": "Inactive"},  
    }  
    customer_info = customers.get(customer_id)  
    if customer_info:  
        # Return customer information in JSON format  
        return json.dumps({  
            "customer_id": customer_id,  
            "membership_level": customer_info["membership_level"],  
            "account_status": customer_info["account_status"]  
        })  
    else:  
        return f"Customer with ID {customer_id} not found."  


# track_shipment_def = {  
#     "name": "track_shipment",  
#     "description": "Provide tracking information for a customer's shipment",  
#     "parameters": {  
#         "type": "object",  
#         "properties": {  
#             "customer_id": {
#                 "type": "string",
#                 "description": "The unique identifier for the customer"
#             },
#             "tracking_number": {  
#                 "type": "string",  
#                 "description": "The tracking number of the shipment"  
#             }  
#         },  
#         "required": ["customer_id", "tracking_number"]  
#     }  
# }  

# # Handler Functions  
# async def track_shipment_handler(customer_id, tracking_number):  
#     statuses = ["In Transit", "Out for Delivery", "Delivered", "Delayed"]  
#     status = random.choice(statuses)  
#     return f"Shipment for customer {customer_id} with tracking number {tracking_number} is currently: {status}."  

# Tools list
tools = [
    (get_customer_info_def, get_customer_info_handler),
    (check_order_status_def, check_order_status_handler),
    (process_return_def, process_return_handler),
    (get_product_info_def, get_product_info_handler),
    (update_account_info_def, update_account_info_handler),
    # (track_shipment_def, track_shipment_handler),  
    (cancel_order_def, cancel_order_handler),  
    (schedule_callback_def, schedule_callback_handler),      
]