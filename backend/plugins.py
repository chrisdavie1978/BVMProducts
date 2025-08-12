# plugins.py
import json
import logging
import httpx
from semantic_kernel.functions import kernel_function
from config import SALSIFY_API_KEY, ORG_ID, MAX_PRODUCTS_TO_SUMMARIZE

class SalsifyProductTool:
    """
    A Semantic Kernel tool to interact with the Salsify Product API.
    """
    def __init__(self, api_key: str, org_id: str):
        self._api_key = api_key
        self._org_id = org_id
        self._http_client = httpx.AsyncClient()

    @kernel_function(
        description="Fetches product data from Salsify based on a URL-encoded filter expression. Returns simplified product data as JSON string for summarization.",
        name="get_products"
    )
    async def get_products(self, filter_query_expression: str) -> str:
        """
        Fetches product data from Salsify API using the provided URL-encoded filter expression.
        The function will prepend 'filter=' to the expression before making the API call.
        It then simplifies the product data to reduce token count for the LLM.
        Returns the simplified JSON response as a string.
        """
        logging.info(f"Salsify Tool: === get_products_tool INVOKED ===")
        logging.info(f"Salsify Tool: Received filter_query_expression: '{filter_query_expression}'")

        parsed_expression = filter_query_expression

        full_filter_param = f"filter={parsed_expression}"

        # Limit per_page to MAX_PRODUCTS_TO_SUMMARIZE to align with prompt instructions
        url = f"https://app.salsify.com/api/v1/orgs/{self._org_id}/products?{full_filter_param}&per_page={MAX_PRODUCTS_TO_SUMMARIZE}"
        headers = {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"}

        logging.info(f"Salsify Tool: Attempting API call to URL: {url}")

        try:
            response = await self._http_client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            product_data = response.json()
            logging.info(f"Json Data: {product_data}")
            # --- OPTIMIZATION: Filter and simplify product data for LLM ---
            simplified_products = []
            # Define keys you care about for the summary
            # Make sure these keys match what's actually in your Salsify API response
            # You might need to inspect actual Salsify JSON output for exact keys (e.g., "name" vs "Product Name")
            # For demonstration, I'm using the friendly names from your prompt, adjust as needed.
            relevant_keys = [
                #'Product Name', 'Item/SKU (Product ID)', 'Finish Code', 'Thickness',
                # Assuming 'main_image' or 'assets' for image URLs
                #'main_image', # Direct image URL field if available
                #'assets' # If assets is a list of objects containing image URLs
                'Item/SKU (Product ID)', 'Item Description', 'Product Name', 'Décor Code', 'Range Description', 'Range Audience', 'Décor Description', 'Global Colour Group', 'Colour Selection 1', 
                'Design Family', 'Design Group', 'Item Description', 'Grade Code', 'Finish Code', 'Size Code', 'Thickness Code'
                'Finish Description', 'Thickness', 'Thickness UOM', 'Item Length', 'Length UOM', 'Item Width','Width UOM'
            ]

            if 'data' in product_data and isinstance(product_data['data'], list):
                for product in product_data['data']: # Loop through all fetched (up to per_page limit)
                    simplified_product = {}
                    for key in relevant_keys:
                        # Direct field mapping
                        if key in product:
                            simplified_product[key] = product[key]
                        # Handling for nested assets for images (common in PIMs)
                        elif key == 'assets' and 'assets' in product and isinstance(product['assets'], list):
                            for asset in product['assets']:
                                # Look for a primary image, adjust 'usage' or 'type' as per Salsify's actual data
                                if asset.get('url') and (asset.get('usage') == 'main_image' or asset.get('type') == 'image'):
                                    simplified_product['main_image_url'] = asset['url'] # Standardize key
                                    break # Found primary image, stop looking in assets
                        # Add more specific mappings if needed (e.g., 'Product Name' might be 'name' in JSON)

                    #image_urls = []
                    #if 'salsify:digital_assets' in product and isinstance(product['salsify:digital_assets'], list):
                        #for asset in product['salsify:digital_assets']:
                            #if 'salsify:url' in asset:
                                #image_urls.append(asset['salsify:url'])
                    
                    #if image_urls:
                        #simplified_product['image_urls'] = image_urls # Add list of URLs

                    simplified_products.append(simplified_product)

            # Prepare the final output for the LLM. It's concise.
            # Only include 'data' and 'meta' to indicate found products/total.
            final_output_for_llm = {
                "data": simplified_products,
                "meta": {
                    "total_results": product_data.get('meta', {}).get('total_results', 0),
                    "current_page": product_data.get('meta', {}).get('current_page', 1)
                }
            }
            # --- END OPTIMIZATION ---

            logging.info(f"Salsify Tool: API Call Successful. Fetched {len(product_data.get('data', []))} products. Simplified to {len(simplified_products)} for LLM.")
            logging.debug(f"Salsify Tool: Simplified API Response (first 500 chars): {str(final_output_for_llm)[:500]}...")

            return json.dumps(final_output_for_llm)
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP Error fetching products (Status {e.response.status_code}): {e.response.text}"
            logging.error(f"Salsify Tool Error: {error_msg}", exc_info=True)
            return json.dumps({"error": error_msg})
        except httpx.RequestError as e:
            error_msg = f"Request Error fetching products: {e}"
            logging.error(f"Salsify Tool Error: {error_msg}", exc_info=True)
            return json.dumps({"error": error_msg})
        except Exception as e:
            error_msg = f"An unexpected error occurred in get_products tool: {e}"
            logging.error(f"Salsify Tool Error: {error_msg}", exc_info=True)
            return json.dumps({"error": error_msg})